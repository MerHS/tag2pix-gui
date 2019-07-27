import math
from PIL import Image, ImageStat, ImageEnhance
from pathlib import Path

def brightness(im):
   stat = ImageStat.Stat(im)
   stat = ImageStat.Stat(im)
   return stat.mean[0]


path = Path('./')
cpath = path / 'liner_colorize'
save_path = path / 'liner_enhance'

for fp in (path / 'liner_deblur').iterdir():
    dimg = Image.open(fp)
    cimg = Image.open(cpath / fp.name)

    cb = brightness(cimg)
    db = brightness(dimg)

    bfact = 1.1 if db == 0 or cb < db else cb / db

    print(bfact)

    
    dd = ImageEnhance.Brightness(dimg)
    dimg = dd.enhance(bfact)

    dd = ImageEnhance.Sharpness(dimg)
    dimg = dd.enhance(1.15)

    dimg.save(save_path / fp.name)