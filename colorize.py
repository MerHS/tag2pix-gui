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

import numpy as np
from PIL import Image

from model.se_resnet import BottleneckX, SEResNeXt
from model.pretrained import se_resnext_half
from loader.dataloader import ColorSpace2RGB

OLD_TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'network_dump', 'tag_dump.pkl')
OLD_NETWORK_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'network_dump', 'tag2pix.pkl')

NEW_TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'network_dump', 'tag_dump.pkl')
NEW_NETWORK_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'network_dump', 'tag2pix_26_epoch.pkl')

PRETRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'network_dump', 'pretrain.pth')

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

gen_cache = None
pret_cache = None
curr_old = True

def colorize(sketch, color_tag_items, gpu=False, is_old=True, input_size=256, layers=[12,8,5,5]):
    global gen_cache, pret_cache, curr_old

    tag_dump = OLD_TAG_FILE_PATH if is_old else NEW_TAG_FILE_PATH
    network_dump = OLD_NETWORK_FILE_PATH if is_old else NEW_NETWORK_FILE_PATH

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

    input_size = 256 if is_old else 512
    resizer = transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS)

    sketch_aug = transforms.Compose([
                    transforms.Pad(pad, padding_mode='reflect'),
                    resizer, 
                    transforms.ToTensor()])

    map_location = "cuda:0" if gpu else "cpu"
    device = torch.device(map_location)

    if is_old != curr_old:
        if is_old:
            from network import Generator
        else:
            from model.new_net import Generator

        pret_cache = se_resnext_half(
            dump_path=PRETRAIN_PATH, 
            num_classes=color_invariant_class_num, 
            input_channels=1)
        gen_cache = Generator(1, output_dim=3, input_size=input_size,
            cv_class_num=color_variant_class_num, 
            iv_class_num=color_invariant_class_num, 
            layers=layers)
    
        pret_cache.eval()
        gen_cache.eval()

        if gpu:
            pret_cache.to(device)
            gen_cache.to(device)

        checkpoint = torch.load(network_dump, map_location=map_location)
        gen_state_dict = gen_cache.state_dict()
        checkpoint_dict = checkpoint if is_old else checkpoint['G']
        checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items() if k[7:] in gen_state_dict}
        gen_state_dict.update(checkpoint_dict)
        gen_cache.load_state_dict(checkpoint_dict)

    color_tag = torch.zeros(1, color_variant_class_num)
    if gpu:
        color_tag = color_tag.to(device)
    
    # print(color_tag_items)
    # print(cv_dict)
    # id_to_name = {v: k for k, v in name_to_id.items()}
    # for k, v in cv_dict.items():
    #     print(id_to_name[k], k)
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

    # if gpu:
    #     Gen = None
    #     Pretrain_ResNeXT = None
    #     torch.cuda.empty_cache()
    
    if w > h:
        crop_pos = (0, math.ceil(input_size * (1 - (h / w))), input_size, input_size)
    else:
        diff = math.ceil(input_size * (1 - (w / h)))
        crop_pos = (diff - (diff // 2), 0, input_size - (diff // 2), input_size)
    
    return img_col.crop(crop_pos)