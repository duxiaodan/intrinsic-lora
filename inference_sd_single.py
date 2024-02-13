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
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    DiffusionPipeline, 
    UNet2DConditionModel, 
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import copy
from PIL import Image
from PIL.ImageOps import exif_transpose
import json
import random
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import center_crop
import matplotlib.cm
from pdb import set_trace
os.environ["HF_HOME"] = "/share/data/2pals/xdu/HF_cache"#Change it to your own Hugginface cache directory

def main():
    set_seed(0)
    task = 'depth' #'normal' # 'depth' # 'albedo' # 'shading'

    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    model_path = Path(f"./pretrained_weights/sd_single_depth_pytorch_model.bin") # Path to LoRA weights
    data_root = Path("./my_dataset") #Path to your RGB data
    output_dir = Path(f"./output./{task}") #Path to your output directory

    os.makedirs(output_dir, exist_ok=True)
    if task == 'normal':
        prompt = 'surface normal'
    elif task == 'depth':
        prompt = 'depth map'
    elif task == 'albedo':
        prompt = 'albedo'
    elif task == 'shading':
        prompt = 'shading'
    else:
        raise NotImplementedError('Not implemented')
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer", 
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder", 
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="vae", 
    )

    scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    max_timestep = scheduler.config.num_train_timesteps
    unet.load_attn_procs(torch.load(model_path))
    unet.to("cuda")
    vae = vae.to("cuda")
    text_encoder = text_encoder.to("cuda") 

    input_files = sorted([p for p in data_root.iterdir() if p.suffix=='.png' or p.suffix=='.jpg'])

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    generator = torch.Generator(device=unet.device).manual_seed(1015)

    for ii, f in enumerate(input_files):
        temp = Image.open(f)
        orig_img = temp
        with torch.inference_mode():
            rgb = image_transforms(orig_img).unsqueeze(0).to('cuda')
            bsz = 1
            timesteps = torch.randint(max_timestep-1, max_timestep, (bsz,), device='cuda')
            timesteps = timesteps.long()
            original_image_embeds = vae.encode(rgb).latent_dist.mode()
            original_image_embeds = original_image_embeds * vae.config.scaling_factor
            text_inputs = tokenize_prompt(tokenizer, prompt).input_ids.to('cuda')
            encoder_hidden_states = text_encoder(text_inputs)[0]
            model_pred = unet(original_image_embeds, timesteps, encoder_hidden_states).sample
            image = vae.decode(model_pred / vae.config.scaling_factor, return_dict=False)[0]
            if task!='depth':
                image = tensor2np(image.clamp(-1.,1.).squeeze())
                val_image = Image.fromarray(image)
            else:
                #Comment out the following lines if you want colorized depth map
                imax = image.max()
                imin = image.min()
                image = (image-imin)/(imax-imin)
                image = image.squeeze().mean(0)
                image = image.cpu().numpy()*255.
                val_image = Image.fromarray(image.astype(np.uint8))
                # Uncomment the following lines if you want colorized depth map
                # image = image.clamp(-1.,1,).squeeze().mean(0)*0.5+0.5
                # val_image = Image.fromarray(colorize(image.cpu()))

            pred_path = output_dir / 'predicted' / f'{f.stem}_{task}.png' 
            pred_path.parent.mkdir(parents=True,exist_ok=True)
            val_image.save(pred_path)

        temp.close()


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

def colorize(value, cmap='inferno'):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = value.squeeze()
    set_trace()
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)
    img = value[...]
    return img[:,:,:3]

if __name__=="__main__":
    main()