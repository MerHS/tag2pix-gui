from pathlib import Path
from PIL import Image
from deblur import deblur, deblur_batch, superpool_img
from tqdm import tqdm
# def main():
#     root = Path('./test-images')
#     res = root / 'res'

#     batch = []
#     batch_fn = []
#     result = []
#     for fn in (root / '5-53').iterdir():
#         img = Image.open(fn)
#         batch.append(img)
#         batch_fn.append(fn.name)

#         if len(batch) == 8:
#             dbr = deblur_batch(batch)
#             result.extend(dbr)
#             batch.clear()
    
#     if len(batch) != 0:
#         dbr = deblur_batch(batch)
#         result.extend(dbr)

#     for name, img in zip(batch_fn, result):
#         img.save(res / name)

# if __name__ == '__main__':
#     main()

def main():
    root = Path('./test-images')
    result = root / 'test_deblur'
    super_save = root / 'test_spp'
    fns = list((root / 'test_colorize').iterdir())
    
    count = 0
    for fn in tqdm(fns):
        if (result / fn.name).exists():
            continue
        img = Image.open(fn)
        skt = Image.open(root / 'simpl_test' / fn.name).convert('RGB')
        spp = superpool_img(img, skt)
        spp.save(super_save / fn.name)

        dbr = deblur(spp)
        dbr.save(result / fn.name)

        count += 1
        if count == 40:
            break

if __name__ == '__main__':
    main()