# original code: https://github.com/bobbens/sketch_simplification
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua
from torch.functional import F
from main import get_resized_img
import numpy as np
from PIL import Image, ImageOps

model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_gan.t7')

def simplify_sketch(pil_img, max_pixel=768, partition=256, padding=16, gpu=False):
   cache = load_lua(model_path, long_size=8)
   model = cache.model
   immean = cache.mean
   imstd  = cache.std
   model.evaluate()

   if gpu:
      model = model.cuda()

   data = pil_img.convert('L')
   data = get_resized_img(data, max_pixel=max_pixel, resample=Image.LANCZOS)
   data = ImageOps.autocontrast(data, ignore=255)
   w, h  = data.size[0], data.size[1]
   pw    = 8-(w%8) if w%8!=0 else 0
   ph    = 8-(h%8) if h%8!=0 else 0
   data  = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
   if pw != 0 or ph != 0:
      data = F.pad(data, (0, pw, 0, ph), mode='reflect').data
   
   simp_arr = _simplify_part(data, model, gpu)

   cache = None
   model = None
   if gpu:
      torch.cuda.empty_cache()
   
   simp_arr = simp_arr[0][:h, :w] * 255
   return Image.fromarray(simp_arr.numpy().astype(np.uint8))

def _simplify_part(img_arr, model, gpu=False):
   
   with torch.no_grad():
      if gpu:
         pred = model.forward(img_arr.cuda()).float().cpu()
      else:
         pred = model.forward(img_arr)

   return pred[0]
