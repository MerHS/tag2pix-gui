import os, pathlib, math
import numpy as np
import cv2

def crop_square_image(img, aspect): # aspect = width / height
    h, w = img.shape[:2] # assume that h == w

    if aspect > 1: # real width > height / crop horizontal letterbox
        cut = math.ceil((h - w / aspect) / 2)
        img = img[cut:-cut, :]
    elif aspect < 1: # real width < height / crop vertical letterbox
        cut = math.ceil((w - h * aspect) / 2)
        img = img[:, cut:-cut]
    return img

def is_white(vect):
    return np.all(vect > 250)

def make_square(img, size=256, crop=False, extend=None):
    """
    extend borders with white pixel or crop image to make it square

    crop: boolean -- crop longer part and make square (default: False)
    extend: None | (boolean, boolean) -- extend (left or top, right or bottom) part to white
    
    return: (image, is_cropped: boolean, is_extended: None | (boolean, boolean))
    """
    is_bgr = len(img.shape) >= 3 and img.shape[2] == 3
    height, width = img.shape[:2]
    
    if crop:
        if height > width: # crop top & bottom
            margin_bottom = (height - width) // 2
            margin_top = (height - width) - margin_bottom
            img = img[margin_top:-margin_bottom]
        elif width > height: # crop left & right
            margin_left = (width - height) // 2
            margin_right = (width - height) - margin_left
            img = img[:, margin_left:-margin_right]

        resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return (resized_img, True, None) 

    elif extend is None:
        if height > width: # extend horizontally
            lw, rw = is_white(img[:, 0]), is_white(img[:, -1])
            extend = (lw, rw)
        elif height < width: # extend vertically
            th, bh = is_white(img[0]), is_white(img[-1])
            extend = (th, bh)
    
    diff = abs(height - width)
    lx, ly = diff // 2, diff - (diff // 2)
    cropped = False

    if height > width: # extend (left, right)
        if extend is not None and extend is not (False, False):
            if is_bgr:
                zx, zy = np.full((height, lx, 3), 255, dtype=np.uint8), np.full((height, ly, 3), 255, dtype=np.uint8)
            else:
                zx, zy = np.full((height, lx), 255, dtype=np.uint8), np.full((height, ly), 255, dtype=np.uint8)

        if extend == (True, True):
            img = np.append(zx, img, axis=1)
            img = np.append(img, zy, axis=1)
        elif extend == (False, True):
            img = np.append(img, zx, axis=1)
            img = np.append(img, zy, axis=1)
        elif extend == (True, False):
            img = np.append(zx, img, axis=1)
            img = np.append(zy, img, axis=1)
        else: # crop top / bottom
            img = img[lx:-ly]
            cropped = True

    elif width > height: # extend (top, bottom)
        if extend is not None and extend is not (False, False):
            if is_bgr:
                zx, zy = np.full((lx, width, 3), 255, dtype=np.uint8), np.full((ly, width, 3), 255, dtype=np.uint8)
            else:
                zx, zy = np.full((lx, width), 255, dtype=np.uint8), np.full((ly, width), 255, dtype=np.uint8)

        if extend == (True, True):
            img = np.append(zx, img, axis=0)
            img = np.append(img, zy, axis=0)
        elif extend == (False, True):
            img = np.append(img, zx, axis=0)
            img = np.append(img, zy, axis=0)
        elif extend == (True, False):
            img = np.append(zx, img, axis=0)
            img = np.append(zy, img, axis=0)
        else: # crop
            img = img[:, lx:-ly]
            cropped = True
    
    resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    return (resized_img, cropped, extend)

    



