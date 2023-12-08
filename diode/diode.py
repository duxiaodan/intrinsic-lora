import os.path as osp
from itertools import chain
import json

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from pathlib import Path

from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import center_crop

'''
The json metadata for DIODE is laid out as follows:
train:
    outdoor:
        scene_000xx:
            scan_00yyy:
                - 000xx_00yyy_indoors_300_010
                - 000xx_00yyy_indoors_300_020
                - 000xx_00yyy_indoors_300_030
        scene_000kk:
            _analogous_
val:
    _analogous_
test:
    _analogous_
'''

_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, )
    for split in tokens:
        assert split in valid_tokens
    return tokens


def enumerate_paths(src):
    '''flatten out a nested dictionary into an iterable
    DIODE metadata is a nested dictionary;
    One could easily query a particular scene and scan, but sequentially
    enumerating files in a nested dictionary is troublesome. This function
    recursively traces out and aggregates the leaves of a tree.
    '''
    if isinstance(src, list):
        return src
    elif isinstance(src, dict):
        acc = []
        for k, v in src.items():
            _sub_paths = enumerate_paths(v)
            _sub_paths = list(map(lambda x: osp.join(k, x), _sub_paths))
            acc.append(_sub_paths)
        return list(chain.from_iterable(acc))
    else:
        raise ValueError('do not accept data type {}'.format(type(src)))


def plot_depth_map(dm, validity_mask):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)

    dm = (dm - np.min(dm)) / np.ptp(dm)
    dm = 1-dm
    dm = np.stack([dm]*3,axis=-1)
    
    dm[np.where(validity_mask == False)] = 0
    dm = Image.fromarray(np.uint8(dm[:,:,:3]*255)).convert('RGB')
    mask = Image.fromarray(np.uint8(validity_mask*255))
    return dm, mask


def plot_normal_map(normal_map):
    normal_viz = normal_map[:, ::, :]

    #Normalize normals
    normi = np.where(np.sum(normal_viz,axis=2)!=0.)
    zero_mask = np.equal(np.sum(normal_viz, 2, keepdims=True), 0.).astype(np.float32)
    linalg_norm = np.sqrt((normal_viz[normi] * normal_viz[normi]).sum(axis=1,keepdims=True))
    normal_viz[normi] = normal_viz[normi]/(linalg_norm+1e-10)
    #Reverse color convention for both Y and Z axis to be consistent with Omnidatav2
    normal_viz[:,:,1:] = normal_viz[:,:,1:]*-1
    #Make masked area [-1,-1,-1]
    normal_viz = normal_viz + zero_mask*(-1)
    normal_viz = (normal_viz +1)/2.
    
    normal_mask = Image.fromarray(np.uint8(255*(normal_viz.sum(2)>0.)))
    normal_viz_img = Image.fromarray(np.uint8(normal_viz*255)).convert('RGB')
    return normal_viz_img, normal_mask
