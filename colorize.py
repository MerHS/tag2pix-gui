import argparse, os, pickle, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import color

import numpy as np
from PIL import Image

from model.se_resnet import BottleneckX, SEResNeXt
from model.pretrained import se_resnext_half
from model.network import Generator

TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tag_dump.pkl')
NETWORK_512_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tag2pix_512.pkl')
NETWORK_256_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tag2pix_256.pkl')
PRETRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth')

gen_cache = None
pret_cache = None
curr_size = None

def get_tag_dict(tag_dump_path):
    cv_dict = dict()
    iv_dict = dict()

    with open(tag_dump_path, 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        name_to_id =  pkl['tag_dict']

    for i, tag_id in enumerate(cv_tag_list):
        cv_dict[tag_id] = i
    for i, tag_id in enumerate(iv_tag_list):
        iv_dict[tag_id] = i

    return iv_dict, cv_dict, name_to_id

class ColorSpace2RGB(object):
    """
    [-1, 1] to [0, 255]
    """
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, img):
        """numpy array [b, [-1~1], [-1~1], [-1~1]] to target space / result rgb[0~255]"""
        img = img.data.numpy()

        if self.color_space == 'rgb':
            img = (img + 1) * 0.5

        img = img.transpose(0, 2, 3, 1)
        if self.color_space == 'lab': # to [0~100, -128~127, -128~127]
            img[:,:,:,0] = (img[:,:,:,0] + 1) * 50
            img[:,:,:,1] = (img[:,:,:,1] * 127.5) - 0.5
            img[:,:,:,2] = (img[:,:,:,2] * 127.5) - 0.5
            img_list = []
            for i in img:
                img_list.append(color.lab2rgb(i))
            img = np.array(img_list)
        elif self.color_space == 'hsv': # to [0~1, 0~1, 0~1]
            img = (img + 1) * 0.5
            img_list = []
            for i in img:
                img_list.append(color.hsv2rgb(i))
            img = np.array(img_list)

        img = (img * 255).astype(np.uint8)
        return img # [0~255] / [b, h, w, 3]

def colorize(sketch, color_tag_items, gpu=False, input_size=256, layers=[12,8,5,5]):
    global gen_cache, pret_cache, curr_size

    tag_dump = TAG_FILE_PATH
    network_dump = NETWORK_256_PATH if input_size == 256 else NETWORK_512_PATH

    sketch = sketch.convert('L')
    iv_dict, cv_dict, name_to_id = get_tag_dict(tag_dump)

    color_variant_class_num = len(cv_dict.keys())
    color_invariant_class_num = len(iv_dict.keys())
    color_revert = ColorSpace2RGB('rgb')

    w, h = sketch.size
    diff = abs(w - h)
    if w > h:
        pad = (0, diff, 0, 0)
    else:
        diff_half = diff // 2
        pad = (diff - diff_half, 0, diff_half, 0)

    sketch_aug = transforms.Compose([
                    transforms.Pad(pad, padding_mode='reflect'),
                    transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS), 
                    transforms.ToTensor()])

    map_location = "cuda:0" if gpu else "cpu"
    device = torch.device(map_location)

    # Load network dump
    if curr_size is not None and input_size != curr_size:
        gen_cache = None
        pret_cache = None
        if gpu:
            torch.cuda.empty_cache()

    if pret_cache is None: 
        pret_cache = se_resnext_half(
            dump_path=PRETRAIN_PATH, 
            num_classes=color_invariant_class_num, 
            input_channels=1)
            
        pret_cache.eval()
        if gpu:
            pret_cache.to(device)
    
    
    if curr_size is None or input_size != curr_size:
        curr_size = input_size
        
        # currently, 512px version does not use bn layer
        if input_size == 512:
            net_opt = {'guide': True, 'relu': False, 'bn': False, 'cit': True}
        else:
            net_opt = {'guide': True, 'relu': False, 'bn': True, 'cit': True}

        gen_cache = Generator(input_size=input_size, output_dim=3, 
            cv_class_num=color_variant_class_num, 
            iv_class_num=color_invariant_class_num, 
            net_opt=net_opt)
    
        gen_cache.eval()
        if gpu:
            gen_cache.to(device)
    
        checkpoint = torch.load(network_dump, map_location=map_location)
        gen_state_dict = gen_cache.state_dict()
        checkpoint_dict = checkpoint['G']
        checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items() if k[7:] in gen_state_dict}
        gen_state_dict.update(checkpoint_dict)
        gen_cache.load_state_dict(checkpoint_dict)

    color_tag = torch.zeros(1, color_variant_class_num)
    if gpu:
        color_tag = color_tag.to(device)
    
    for line in color_tag_items:
        l = line.strip()
        cv_id = name_to_id[l]
        color_tag[:, cv_dict[cv_id]] = 1

    img_col = None
    with torch.no_grad():
        sketch_ = sketch_aug(sketch).unsqueeze(0)
        if gpu:
            sketch_ = sketch_.to(device)

        feature_tensor = pret_cache(sketch_) 
        if gpu:
            feature_tensor = feature_tensor.to(device)

        G_col, _ = gen_cache(sketch_, feature_tensor, color_tag)

        if gpu:
            G_col = G_col.cpu()

        G_col = color_revert(G_col)
        img_col = Image.fromarray(G_col[0]) 


    if w > h:
        crop_pos = (0, math.ceil(input_size * (1 - (h / w))), input_size, input_size)
    else:
        diff = math.ceil(input_size * (1 - (w / h)))
        crop_pos = (diff - (diff // 2), 0, input_size - (diff // 2), input_size)
    
    return img_col.crop(crop_pos)
