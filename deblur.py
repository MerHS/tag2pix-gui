import tensorflow as tf
import numpy as np
import cv2
import time, os

from PIL import Image
from skimage import measure
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from model.deblur import DEBLUR

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_segment(img):
    return felzenszwalb(img, scale=50, sigma=0.8, min_size=30) + 1

def paint_mean_segment(segments, img):
    new_img = np.zeros(list(img.shape), dtype=np.uint8)
    for channel in range(img.shape[2]):
        regions = measure.regionprops(segments, intensity_image=img[:,:,channel])
        for r in regions:
            m_intens = int(r.mean_intensity)
            rp = r.coords
            for i in range(rp.shape[0]):
                new_img[rp[i][0]][rp[i][1]][channel] = m_intens

    return new_img

def superpool_img(rgb_img, skt_img, seg_by_sketch=True):
    rgb_img = np.array(rgb_img)
    skt_img = np.array(skt_img)

    rgb_img = cv2.resize(rgb_img, (256, 256))
    skt_img = cv2.resize(skt_img, (256, 256))

    if seg_by_sketch:
        rgb_seg = get_segment(skt_img)
    else:
        rgb_seg = get_segment(rgb_img)

    sg_avg = paint_mean_segment(rgb_seg, rgb_img)
    merge_img = ((sg_avg / 255.) * (skt_img / 255.) * 255).astype(np.uint8)

    return Image.fromarray(merge_img)

class DeblurArgs():
    def __init__(self, use_gpu):
        self.phase = 'test'
        self.datalist = 'test'
        self.batch_size = 1
        self.model = 'color'
        self.epoch = 1
        self.lr = 1e-4
        self.gpu = 0 if use_gpu else -1
        self.height = 256
        self.width = 256

def deblur(img, use_gpu, test_dir):
    '''img: cv2 image / return cv2 img (scipy.misc.im)'''

    args = DeblurArgs(use_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    deblur = DEBLUR(args)

    return deblur.test(img, 256, 256)