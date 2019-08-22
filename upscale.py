import os
from pathlib import Path
from PIL import Image

W2X_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'waifu2x-caffe')

w2x_path = Path(W2X_ROOT)
w2x_exe = w2x_path / 'waifu2x-caffe-cui.exe'
temp_path = w2x_path / 'temp'


def upscale(img, gpu, height):
    if not w2x_path.exists() and not w2x_exe.exists():
        return "waifu does not exists."

    in_path = str(temp_path / 'temp.png')
    out_path = str(temp_path / 'upscale.png')

    img.save(in_path)

    g_opt = '-p gpu' if gpu else '-p cpu'
    h_opt = f'-h {height}'
    os.system(f'{str(w2x_exe)} -i {in_path} {g_opt} {h_opt} -o {out_path}')

    return Image.open(out_path).convert('RGB')
