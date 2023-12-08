# coding=utf-8
# Intrinsic-LoRA
"""Intrinsic-LoRA AugUNet model for normal training"""

import argparse
import logging
import math
import os
import os.path as osp
import random
import shutil
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import copy
from PIL import Image
from PIL.ImageOps import exif_transpose
from diode.diode import (
    plot_normal_map,
    check_and_tuplize_tokens,
    enumerate_paths,
    _VALID_SPLITS, 
    _VALID_SCENE_TYPES
)
import json
import datetime
#Xiaodan: according to https://arxiv.org/pdf/2305.08891.pdf
from rescale_cfg_pipeline_forward import new_call

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)
def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length
    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs

def tensor2np(tensor):
    return (255*(tensor.cpu().permute(1,2,0).numpy()*0.5+0.5)).astype(np.uint8)

def listPILToTensor(listPILs):
    size = listPILs[0].size[0]
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return torch.stack([image_transforms(p) for p in listPILs])

def visualization_routine(gt,im_1,im_2,im_3):
    gt = tensor2np(gt)
    im_1 = tensor2np(im_1)
    im_2 = tensor2np(im_2)
    im_3 = np.array(im_3)
    return Image.fromarray(np.hstack((im_1,gt,im_2,im_3)))


@torch.inference_mode()
def log_validation(
    text_encoder, 
    tokenizer, 
    unet, 
    vae, 
    args, 
    accelerator,
    zero_snr_betas,
    test_batches, 
    train_batch, 
    weight_dtype, 
    epoch, 
    global_step
):
    pipeline = DiffusionPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )

    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, **scheduler_args,
        #Xiaodan: according to https://arxiv.org/pdf/2305.08891.pdf
        prediction_type='v_prediction',
        trained_betas=zero_snr_betas,
    )
    assert pipeline.scheduler.prediction_type == "v_prediction"
    assert pipeline.scheduler.alphas_cumprod[-1] == 0.

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.new_call = new_call.__get__(pipeline, pipeline.__class__)


    val_test_images1 = []
    val_test_images2 = []
    val_train_images = []
    test_batch1 = test_batches[0]
    test_batch2 = test_batches[1]
    for ii in range(4):
        with torch.no_grad():
            if test_batches[0] is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed+100) if args.seed else None
                image = pipeline.new_call(
                    prompt_embeds = pipeline.text_encoder(test_batch1['input_ids'][ii:ii+1])[0],
                    image = Image.fromarray(tensor2np(test_batch1['original_pixel_values'][ii])),
                    image_guidance_scale = 1.,
                    guidance_scale = 3.0,
                    generator=generator,
                    num_inference_steps = 25,
                    #Xiaodan: according to https://arxiv.org/pdf/2305.08891.pdf
                    guidance_rescale = 0.7,
                ).images[0]
                val_test_images1.append(image)
            if test_batches[1] is not None:
                image = pipeline.new_call(
                    prompt_embeds = pipeline.text_encoder(test_batch2['input_ids'][ii:ii+1])[0],
                    # latents=test_batch2['noises'][ii].unsqueeze(0),
                    image = Image.fromarray(tensor2np(test_batch2['original_pixel_values'][ii])),
                    image_guidance_scale = 1.,
                    guidance_scale = 3.0,
                    generator=generator,
                    num_inference_steps = 25,
                    #Xiaodan: according to https://arxiv.org/pdf/2305.08891.pdf
                    guidance_rescale = 0.7,
                ).images[0]
                val_test_images2.append(image)
            
            val_train_image = pipeline.new_call(
                prompt_embeds=pipeline.text_encoder(train_batch['input_ids'][ii:ii+1])[0],
                # latents=train_batch['noises'][ii].unsqueeze(0),
                image = Image.fromarray(tensor2np(train_batch['original_pixel_values'][ii])),
                image_guidance_scale = 1.,
                guidance_scale = 3.0,
                generator = generator,
                num_inference_steps = 25,
                #Xiaodan: according to https://arxiv.org/pdf/2305.08891.pdf
                guidance_rescale = 0.7,
            ).images[0]
            val_train_images.append(val_train_image)
    concat_test_images1 = []
    concat_test_images2 = []
    concat_train_images = []
    for gt, im_1, im_2, im_3 in zip(test_batch1['gt_values'],test_batch1['original_pixel_values'],test_batch1['pixel_values'],val_test_images1):
        output_img = visualization_routine(gt,im_1,im_2,im_3)
        concat_test_images1.append(output_img)

    for gt, im_1, im_2, im_3 in zip(test_batch2['gt_values'],test_batch2['original_pixel_values'],test_batch2['pixel_values'],val_test_images2):
        output_img = visualization_routine(gt,im_1,im_2,im_3)
        concat_test_images2.append(output_img)

    for gt, im_1, im_2, im_3 in zip(train_batch['gt_values'],train_batch['original_pixel_values'],train_batch['pixel_values'],val_train_images):
        output_img = visualization_routine(gt,im_1,im_2,im_3)
        concat_train_images.append(output_img)

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation: training images": [
                        wandb.Image(image, )
                        for i, image in enumerate(concat_train_images)
                    ],
                }, 
                step=global_step
            )
            tracker.log(
                {
                    "validation: test images 1": [
                        wandb.Image(image, )
                        for i, image in enumerate(concat_test_images1)
                    ],
                }, 
                step=global_step
            )

            tracker.log(
                {
                    "validation: test images 2": [
                        wandb.Image(image, )
                        for i, image in enumerate(concat_test_images2)
                    ],
                }, 
                step=global_step
            )
    del pipeline
    torch.cuda.empty_cache()
    return
class PSEUDODataset(Dataset):
    def __init__(
        self,
        data_root,
        pseudo_root,
        tokenizer,
        splits,
        scene_types,
        size=512,
        center_crop=True,
        num_train_imgs=None,
        tokenizer_max_length=None,
        empty_prompt = False,
        unified_prompt = None,
    ):
        self.data_root = Path(data_root)
        self.pseudo_root = Path(pseudo_root)
        self.splits = check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )
        meta_fname = self.data_root.parent / 'diode_meta.json'
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.num_train_imgs = num_train_imgs
        self.empty_prompt = empty_prompt
        self.unified_prompt = unified_prompt
        if not self.data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        imgs = []
        for split in self.splits:
            for scene_type in self.scene_types:
                _curr = enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: osp.join(split, scene_type, x), _curr)
                _curr = list(_curr)
                for curr_im in _curr:
                    if Path(self.data_root / f'{curr_im}_normal.npy').exists():
                        imgs.append(curr_im)
        if self.num_train_imgs is not None and self.num_train_imgs<len(imgs):
            #Subsample imgs and captions with same indices
            imgs = random.sample(imgs, self.num_train_imgs)
        self.imgs = imgs

        self._length = len(self.imgs)
        print(f'Dataset length is {self._length}')

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        im = self.imgs[index%self._length]
        example = {}
        instance_image = np.load(self.data_root / f'{im}_normal.npy').squeeze()
        instance_image,_ = plot_normal_map(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        rgb_image = Image.open(self.data_root.parent / 'depths' / f'{im}.png')
        rgb_image = exif_transpose(rgb_image)
        if not rgb_image.mode == "RGB":
            rgb_image = rgb_image.convert("RGB")

        pseudo_image = Image.open(self.pseudo_root / f'{im}.png')
        pseudo_image = exif_transpose(pseudo_image)
        if not pseudo_image.mode == "RGB":
            pseudo_image = pseudo_image.convert("RGB")

        if self.empty_prompt:
            prompt = ""
        if self.unified_prompt is not None:
            prompt = self.unified_prompt
        text_inputs = tokenize_prompt(
            self.tokenizer, prompt, tokenizer_max_length=self.tokenizer_max_length
        )

        example["instance_images"] = self.image_transforms(pseudo_image)
        example["gt_images"] = self.image_transforms(instance_image)
        example["original_instance_images"] = self.image_transforms(rgb_image)
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask
        return example

def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]

    pixel_values = [example["instance_images"] for example in examples]
    gt_values = [example["gt_images"] for example in examples]
    original_pixel_values = [example["original_instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()    
    
    gt_values = torch.stack(gt_values)
    gt_values = gt_values.to(memory_format=torch.contiguous_format).float()

    original_pixel_values = torch.stack(original_pixel_values)
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "original_pixel_values": original_pixel_values,
        "gt_values": gt_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='v_prediction',
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--empty_prompt", 
        action="store_true", 
        help="If turned on, inputting empty string as prompt during training"
    )
    parser.add_argument(
        "--unified_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_train_imgs",
        type=int,
        default=None,
    )
    parser.add_argument(
        '--scene_types',
        type=str,
        default='indoors',
        help="'indoors','outdoor' or 'outdoor,indoors'"
    )
    parser.add_argument(
        '--pseudo_root',
        type=str,
        required=True,
        help="root of peudo labels that are used during training"
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

def main():
    stop_time = datetime.datetime.now() + datetime.timedelta(hours=3.9)
    args = parse_args()
    #Xiaodan: according to https://arxiv.org/pdf/2305.08891.pdf
    assert args.prediction_type == "v_prediction"
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=bool(args.resume_from_checkpoint))

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    #Xiaodan: changed according to https://arxiv.org/pdf/2305.08891.pdf
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler",
        rescale_betas_zero_snr=True, 
        timestep_spacing="trailing",
        prediction_type='v_prediction'
    )
    assert noise_scheduler.alphas_cumprod[-1] == 0.
    zero_snr_betas = noise_scheduler.betas
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="unet", revision=args.revision
    )
    ip2p_conv_in = copy.deepcopy(unet.conv_in)
    del unet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    sd_conv_in = copy.deepcopy(unet.conv_in)

    new_weight = torch.empty_like(ip2p_conv_in.weight.data)
    new_bias = torch.empty_like(ip2p_conv_in.bias.data)

    new_weight[:,:4,:,:] = sd_conv_in.weight.data
    new_weight[:,4:,:,:] = ip2p_conv_in.weight.data[:,4:,:,:]
    new_bias[:] = sd_conv_in.bias.data

    unet.conv_in = ip2p_conv_in
    unet.conv_in.weight = torch.nn.Parameter(new_weight.detach())
    unet.conv_in.bias = torch.nn.Parameter(new_bias.detach())
    unet.config.in_channels = 8
    del ip2p_conv_in, sd_conv_in, new_weight, new_bias
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}

    names = []
    for name in unet.attn_processors.keys():
        names.append(name)
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    layers_to_update = unet.attn_processors
    attn_lora_layers = AttnProcsLayers(layers_to_update)

    learnable_params = attn_lora_layers.parameters()
    input_to_optim = learnable_params
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        input_to_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = PSEUDODataset(
        data_root = args.train_data_dir,
        pseudo_root = args.pseudo_root,
        size=512,
        splits='train',
        scene_types=args.scene_types.split(','),
        center_crop=True,
        num_train_imgs = args.num_train_imgs,
        tokenizer = tokenizer,
        tokenizer_max_length=None,
        empty_prompt = args.empty_prompt,
        unified_prompt = args.unified_prompt,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader2 = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        batch_size=4,
        num_workers=args.dataloader_num_workers,
    )
    test_dataset1 = PSEUDODataset(
        data_root = args.train_data_dir,
        pseudo_root = args.pseudo_root,
        size=512,
        splits='val',
        scene_types=('indoors'),
        center_crop=True,
        num_train_imgs = 100000000000,
        tokenizer = tokenizer,
        tokenizer_max_length=None,
        empty_prompt = args.empty_prompt,
        unified_prompt = args.unified_prompt,
    )
    test_dataloader1 = torch.utils.data.DataLoader(
        test_dataset1,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        batch_size=4,
        num_workers=args.dataloader_num_workers,
    )
    test_dataset2 = PSEUDODataset(
        data_root = args.train_data_dir,
        pseudo_root = args.pseudo_root,
        size=512,
        splits='val',
        scene_types=('outdoor'),
        center_crop=True,
        num_train_imgs = 1000000000000,
        tokenizer = tokenizer,
        tokenizer_max_length=None,
        empty_prompt = args.empty_prompt,
        unified_prompt = args.unified_prompt,
    )
    test_dataloader2 = torch.utils.data.DataLoader(
        test_dataset2,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        batch_size=4,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.

    attn_lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        attn_lora_layers, optimizer, train_dataloader, lr_scheduler
    )    

    train_dataloader2 = accelerator.prepare(train_dataloader2)
    train_batch = next(iter(train_dataloader2))
    del train_dataloader2

    test_dataloader1 = accelerator.prepare(test_dataloader1)
    test_batch1 = next(iter(test_dataloader1))
    del test_dataset1
    del test_dataloader1

    test_dataloader2 = accelerator.prepare(test_dataloader2)
    test_batch2 = next(iter(test_dataloader2))
    del test_dataset2
    del test_dataloader2
    test_batches = [test_batch1, test_batch2]
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    global_step = 0
    first_epoch = 0
    args.initial_resumed = False
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            args.initial_resumed = True
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.initial_resumed:
        wandb_id = wandb.util.generate_id()
        with open(Path(args.output_dir,'wandb_id.txt'),'a+') as id_f:
            id_f.write(wandb_id)
    elif args.resume_from_checkpoint:
        with open(Path(args.output_dir,'wandb_id.txt'),'r') as id_f:
             wandb_id = id_f.readlines()[0].strip()
    else:
        wandb_id = None
    args.trial_name = Path(args.output_dir).stem
    init_kwargs={
        "wandb":{
            "name":args.trial_name,
            'id': wandb_id,
            'resume':args.initial_resumed or args.resume_from_checkpoint,
        }
    }
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        if not args.resume_from_checkpoint:
            accelerator.init_trackers("intrinsic-lora", config=tracker_config, init_kwargs=init_kwargs)
        else:
            accelerator.init_trackers("intrinsic-lora", init_kwargs=init_kwargs)
            wandb.config.update(tracker_config,allow_val_change=True)    

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    broken=False
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            if global_step == 0 and step==0:
                log_validation(
                    text_encoder, tokenizer, unet, vae, args, accelerator, zero_snr_betas,
                    test_batches, train_batch, weight_dtype, epoch, global_step
                )
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                assert noise_scheduler.config.prediction_type == "v_prediction"
                if noise_scheduler.config.prediction_type == "epsilon":
                    raise ValueError(f'To fix color shift, prediction type needs to be v_prediction')
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    raise NotImplementedError("This code is not tested for snr_gamma")
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = learnable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0:
                    log_validation(
                        text_encoder, tokenizer, unet, vae, args, accelerator, zero_snr_betas,
                        test_batches, train_batch, weight_dtype, epoch, global_step
                    )

            if global_step >= args.max_train_steps:
                break
            if datetime.datetime.now() > stop_time:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                broken=True
                break
        if broken:
            break
            

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()