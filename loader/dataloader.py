import cv2
import pickle, random
import torch
import math, time
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import color
from torchvision import transforms, datasets
from torchvision.transforms import functional as tvF
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets
from .xdog_blend import blend_xdog_and_sketch, add_intensity

TAG_SIMPLE_BACKGROUND = 412368
TAG_WHITE_BACKGROUND = 515193
TAG_SOLO = 212816

def pseudo_uniform(id, a, b):
    return (((id * 1.253 + a * 324.2351 + b * 534.342) * 20147.2312369804) + 0.12949) % (b - a) + a

def real_uniform(id, a, b):
    return random.uniform(a, b)

def get_tagname_from_tensor(id_to_name, iv_tensor, cv_tensor, iv_dict, cv_dict, save_path):
    iv_tag_name_list = []
    cv_tag_name_list = []
    iv_dict_inverse =  {y:x for x,y in iv_dict.items()}
    cv_dict_inverse =  {y:x for x,y in cv_dict.items()}
    f = open(save_path, 'w')
    f.write("iv tags\n")

    for batch_unit in iv_tensor:
        tag_list = [] 
        for i, is_tag in enumerate(batch_unit):
            if is_tag == 1.0:
                tag_name = id_to_name[iv_dict_inverse[i]]
                tag_list.append(tag_name)
                f.write(tag_name + " ")
        iv_tag_name_list.append(tag_list)
        f.write("\n")

    f.write("cv tags\n")
    
    for batch_unit in cv_tensor:
        tag_list = [] 
        for i, is_tag in enumerate(batch_unit):
            if is_tag == 1.0:
                tag_name = id_to_name[cv_dict_inverse[i]]
                tag_list.append(id_to_name[cv_dict_inverse[i]])
                f.write(tag_name + " ")
        cv_tag_name_list.append(tag_list)        
        f.write("\n")

    # print(iv_tag_name_list, cv_tag_name_list)
    f.close()

    return iv_tag_name_list, cv_tag_name_list

def get_classid_dict(tag_dump_path, is_weight, weight_value):
    cv_dict = dict()
    iv_dict = dict()
    id_to_name = dict()
    try:
        f = open(tag_dump_path, 'rb')
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        name_to_id =  pkl['tag_dict']
        id_to_count = pkl['tag_count_dict']

    except EnvironmentError:
        raise Exception(f'{tag_dump_path} does not exist. You should make tag dump file using taglist/tag_indexer.py')
    
    for i, tag_id in enumerate(iv_tag_list):
        iv_dict[tag_id] = i

    for i, tag_id in enumerate(cv_tag_list):
        cv_dict[tag_id] = i

    id_to_name = {y:x for x,y in name_to_id.items()}

    if is_weight:
        iv_class_len = len(iv_dict.keys())
        cv_class_len = len(cv_dict.keys())
        iv_weight_class = torch.zeros(iv_class_len, dtype=torch.float)
        cv_weight_class = torch.zeros(cv_class_len, dtype=torch.float)

        for i, tag_id in enumerate(iv_tag_list):
            iv_weight_class[i] = 1/id_to_count[str(tag_id)]

        for i, tag_id in enumerate(cv_tag_list):
            cv_weight_class[i] = 1/id_to_count[str(tag_id)] 

        # norm to mean = 5, std = 1
        # iv_weight_class = (iv_weight_class-iv_weight_class.mean())/(iv_weight_class.std()) + 5
        # cv_weight_class = (cv_weight_class-cv_weight_class.mean())/(cv_weight_class.std()) + 5

        # norm to [0,1]
        min_iv = torch.min(iv_weight_class)
        range_iv = torch.max(iv_weight_class) - min_iv

        if range_iv > 0:
            iv_weight_class = weight_value[0] * (iv_weight_class - min_iv) / range_iv
        else:
            iv_weight_class = torch.zeros(iv_class_len)

        min_cv = torch.min(cv_weight_class)
        range_cv = torch.max(cv_weight_class) - min_cv

        if range_cv > 0:
            cv_weight_class = weight_value[1] * (cv_weight_class - min_cv) / range_cv
        else:
            cv_weight_class = torch.zeros(cv_class_len)

        print("iv loss weight is %d, cv loss weight is %d" % (weight_value[0], weight_value[1]))
        # print(iv_weight_class, cv_weight_class)
        return (iv_dict, cv_dict, id_to_name, iv_weight_class, cv_weight_class)
    else:
        return (iv_dict, cv_dict, id_to_name, None, None)


def read_tagline_txt(tag_txt_path, img_dir_path, iv_dict, cv_dict, data_size=0, is_train=True):
    iv_class_len = len(iv_dict)
    cv_class_len = len(cv_dict)
    print("read_tagline_txt! We will use %d, %d tags" % (iv_class_len, cv_class_len))
    # tag one-hot encoding + 파일 있는지 확인
    if not tag_txt_path.exists():
        raise Exception(f'tag list text file "{tag_txt_path}" does not exist.')

    iv_tag_set = set(iv_dict.keys())
    cv_tag_set = set(cv_dict.keys())
    iv_class_list = []
    cv_class_list = []
    file_id_list = []

    data_limited = data_size != 0
    count = 0
    count_all = 0
    all_tag_num = 0
    awful_tag_num = 0
    iv_tag_num = 0
    cv_tag_num = 0

    include_tags = [470575, 540830]
    hair_tags = [87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534]
    eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186]

    tag_lines = []
    with tag_txt_path.open('r') as f:
        for line in f:
            tag_lines.append(line)

    random.seed(10)
    random.shuffle(tag_lines)
    random.seed(time.time())
    
    for line in tag_lines:
        count_all += 1
        tag_list = list(map(int, line.split(' ')))
        file_id = tag_list[0]
        tag_list = set(tag_list[1:])

        if not (img_dir_path / f'{file_id}.png').exists():
            continue

        # For face images
        # if len(tag_list) < 8:
        #     continue

        # one girl or one boy / one hair and eye color
        person_tag = tag_list.intersection(include_tags)
        hair_tag = tag_list.intersection(hair_tags)
        eye_tag = tag_list.intersection(eye_tags)

        if not (len(hair_tag) == 1 and len(eye_tag) == 1 and len(person_tag) == 1):
            # print(file_id, hair_tag, eye_tag)
            awful_tag_num += 1
            continue
        
        iv_class = torch.zeros(iv_class_len, dtype=torch.float)
        cv_class = torch.zeros(cv_class_len, dtype=torch.float)
        tag_exist = False

        for tag in tag_list:
            if tag in iv_tag_set:
                try:
                    iv_class[iv_dict[tag]] = 1
                    tag_exist = True
                    iv_tag_num += 1
                except IndexError as e:
                    print(len(iv_dict), iv_class_len, tag, iv_dict[tag])
                    raise e

        if not tag_exist:
            continue
        tag_exist = False

        for tag in tag_list:
            if tag in cv_tag_set:
                try:
                    cv_class[cv_dict[tag]] = 1
                    tag_exist = True
                    cv_tag_num += 1
                except IndexError as e:
                    print(len(cv_dict), cv_class_len, tag, cv_dict[tag])
                    raise e

        if not tag_exist:
            continue

        file_id_list.append(file_id)
        iv_class_list.append(iv_class)
        cv_class_list.append(cv_class)

        all_tag_num += len(tag_list)
        count += 1
        if data_limited and count > data_size:
            break

    print(f'count_all {count_all}, select_count {count}, awful_count {awful_tag_num}, all_tag_num {all_tag_num}, iv_tag_num {iv_tag_num}, cv_tag_num {cv_tag_num}')
    return (file_id_list, iv_class_list, cv_class_list)


class ColorAndSketchDataset(Dataset):
    def __init__(self, rgb_path, sketch_path_list, file_id_list, iv_class_list, cv_class_list,
            override_len=None, both_transform=None, sketch_transform=None, color_transform=None, **kwargs):

        self.rgb_path = rgb_path
        self.sketch_path_list = sketch_path_list
        
        self.file_id_list = file_id_list # copy

        self.iv_class_list = iv_class_list
        self.cv_class_list = cv_class_list

        self.both_transform = both_transform
        self.color_transform = color_transform
        self.sketch_transform = sketch_transform
        self.data_len = len(file_id_list)

        if override_len > 0 and self.data_len > override_len:
            self.data_len = override_len
        self.idx_shuffle = list(range(self.data_len))
        random.seed(10)
        random.shuffle(self.idx_shuffle)
        random.seed(time.time())

    def __getitem__(self, idx):
        index = self.idx_shuffle[idx]

        file_id = self.file_id_list[index]
        iv_tag_class = self.iv_class_list[index]
        cv_tag_class = self.cv_class_list[index]

        sketch_path = random.choice(self.sketch_path_list)
        color_path = self.rgb_path / f"{file_id}.png"
        sketch_path = sketch_path / f"{file_id}.png"

        color_img = Image.open(color_path).convert('RGB') 
        sketch_img = Image.open(sketch_path).convert('L')  # to [1, H, W]
        
        if self.both_transform is not None:
            color_img, sketch_img = self.both_transform(color_img, sketch_img)
        if self.color_transform is not None:
            color_img = self.color_transform(color_img)
        if self.sketch_transform is not None:
            sketch_img = self.sketch_transform(sketch_img)

        return (color_img, sketch_img, iv_tag_class, cv_tag_class)

    def __len__(self):
        return self.data_len

class RGB2ColorSpace(object):
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, img):
        if self.color_space == 'rgb':
            return (img * 2 - 1.)

        img = img.permute(1, 2, 0) # to [H, W, 3]
        if self.color_space == 'lab':
            img = color.rgb2lab(img) # [0~100, -128~127, -128~127]
            img[:,:,0] = (img[:,:,0] - 50.0) * (1 / 50.)
            img[:,:,1] = (img[:,:,1] + 0.5) * (1 / 127.5)
            img[:,:,2] = (img[:,:,2] + 0.5) * (1 / 127.5)
        elif self.color_space == 'hsv':
            img = color.rgb2hsv(img) # [0~1, 0~1, 0~1]
            img = (img * 2 - 1)
        # elif self.color_space == 'yuv':
        #     img = color.rgb2yuv(img) # Maybe [0~1, -0.436~0.436, -0.615~0.615] 
        #     img[:,:,0] = img[:,:,0] * 2 - 1

        # to [3, H, W]
        return torch.from_numpy(img).float().permute(2, 0, 1) # [-1~1, -1~1, -1~1]

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
        # elif self.color_space == 'yuv':
        #     img = color.rgb2yuv(img) # Maybe [0~1, -0.436~0.436, -0.615~0.615]
        #     img =  

        img = (img * 255).astype(np.uint8)
        return img # [0~255] / [b, h, w, 3]


def rot_crop(x):
    """return maximum width ratio of rotated image without letterbox"""
    x = abs(x)
    deg45 = math.pi * 0.25
    deg135 = math.pi * 0.75
    x = x * math.pi / 180
    a = (math.sin(deg135 - x) - math.sin(deg45 - x))/(math.cos(deg135-x)-math.cos(deg45-x))
    return math.sqrt(2) * (math.sin(deg45-x) - a*math.cos(deg45-x)) / (1-a)

class RandomFRC(transforms.RandomResizedCrop):
    """RandomHorizontalFlip + RandomRotation + RandomResizedCrop 2 images"""
    def __call__(self, img1, img2):
        if random.random() < 0.5:
            img1 = tvF.hflip(img1)
            img2 = tvF.hflip(img2)
        if random.random() < 0.5:
            rot = random.uniform(-10, 10)
            crop_ratio = rot_crop(rot)
            img1 = tvF.rotate(img1, rot, resample=Image.BILINEAR)
            img2 = tvF.rotate(img2, rot, resample=Image.BILINEAR)
            img1 = tvF.center_crop(img1, int(img1.size[0] * crop_ratio))
            img2 = tvF.center_crop(img2, int(img2.size[0] * crop_ratio))
        
        i, j, h, w = self.get_params(img1, self.scale, self.ratio)
        #i2, j2, h2, w2 = self.get_params(img2, self.scale, self.ratio)
        # return the image with the same transformation
        return (tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation), 
                tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation))


def get_dataset(args):
    data_dir_path = Path(args.data_dir)
    
    batch_size = args.batch_size
    input_size = args.input_size
    
    data_randomize = RandomFRC(512, scale=(0.9, 1.0), ratio=(0.95, 1.05))

    swap_color_space = [RGB2ColorSpace(args.color_space)] 
    random_jitter = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    data_augmentation = [transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS), 
                        transforms.ToTensor()]
    
    iv_dict, cv_dict, id_to_name, _, _ = get_classid_dict(args.tag_dump, False, [])
    iv_class_len = len(iv_dict.keys())
    cv_class_len = len(cv_dict.keys())

    print('reading tagline')
    data_size = args.data_size
    rgb_train_path = data_dir_path / "rgb_train"
    rgb_test_path = data_dir_path / "benchmark"
    sketch_dir_path_list = ["keras_train", "simpl_train", "xdog_train"]
    sketch_dir_path_list = list(map(lambda x : data_dir_path / x, sketch_dir_path_list))
    sketch_train_path = data_dir_path / "keras_test"
    tag_path = data_dir_path / "tags.txt"

    (train_id_list, train_iv_class_list, train_cv_class_list) = read_tagline_txt(
        tag_path, rgb_train_path, iv_dict, cv_dict, data_size=data_size)

    (test_id_list, test_iv_class_list, test_cv_class_list) = read_tagline_txt(
        tag_path, rgb_test_path, iv_dict, cv_dict, data_size=100)

    print('making train set...')

    train = ColorAndSketchDataset(rgb_path=rgb_train_path, sketch_path_list=sketch_dir_path_list, 
        file_id_list=train_id_list, iv_class_list=train_iv_class_list, cv_class_list=train_cv_class_list, 
        override_len=data_size, both_transform=data_randomize, 
        sketch_transform=transforms.Compose(random_jitter + data_augmentation),
        color_transform=transforms.Compose(data_augmentation + swap_color_space))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.thread)
    
    print(f'iv_class_len={iv_class_len}, cv_class_len={cv_class_len}')
    print(f'train: read {sketch_dir_path_list[0]}, id_list len={len(train_id_list)}, iv_class len={len(train_iv_class_list)}, cv_class len={len(train_cv_class_list)}')

    print('making test set...')
    test = ColorAndSketchDataset(rgb_path=rgb_test_path, sketch_path_list=[sketch_train_path], 
        file_id_list=test_id_list, iv_class_list=test_iv_class_list, cv_class_list=test_cv_class_list,
        override_len=100, both_transform=data_randomize, 
        sketch_transform=transforms.Compose(data_augmentation),
        color_transform=transforms.Compose(data_augmentation + swap_color_space))

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread)
    
    print(f'test: read {sketch_train_path}, id_list len={len(test_id_list)}, iv_class len={len(test_iv_class_list)}, cv_class len={len(test_cv_class_list)}')
   
    return train_loader, test_loader


class SuperpixelDataset(Dataset):
    def __init__(self, rgb_path, pixel_path_list, sketch_path_list, file_id_list, override_len=None, 
            crop_transform=None, rgb_transform=None, sketch_transform=None, **kwargs):

        self.rgb_path = rgb_path
        self.pixel_path_list = pixel_path_list
        self.sketch_path_list = sketch_path_list
        
        self.file_id_list = file_id_list # copy

        self.crop_transform = crop_transform
        self.rgb_transform = rgb_transform
        self.sketch_transform = sketch_transform

        self.data_len = len(file_id_list)

        if override_len > 0 and self.data_len > override_len:
            self.data_len = override_len
        self.idx_shuffle = list(range(self.data_len))
        random.seed(10)
        random.shuffle(self.idx_shuffle)
        random.seed(time.time())

    def __getitem__(self, idx):
        index = self.idx_shuffle[idx]

        file_id = self.file_id_list[index]

        pixel_path = random.choice(self.pixel_path_list)
        sketch_path = random.choice(self.sketch_path_list)
        color_path = self.rgb_path / f"{file_id}.png"
        pixel_path = pixel_path / f"{file_id}.png"
        sketch_path = sketch_path / f"{file_id}.png"

        color_img = Image.open(color_path).convert('RGB') 
        pixel_img = Image.open(pixel_path).convert('RGB')
        sketch_img = Image.open(sketch_path).convert('L')  # to [1, H, W]
        
        if self.crop_transform is not None:
            color_img, sketch_img, pixel_img = self.crop_transform(color_img, sketch_img, pixel_img)
        if self.rgb_transform is not None:
            color_img = self.rgb_transform(color_img)
            pixel_img = self.rgb_transform(pixel_img)
        if self.sketch_transform is not None:
            sketch_img = self.sketch_transform(sketch_img)

        return (color_img, sketch_img, pixel_img)

    def __len__(self):
        return self.data_len

class RandomFRC3(transforms.RandomResizedCrop):
    """RandomHorizontalFlip + RandomRotation + RandomResizedCrop 3 images"""
    def __call__(self, img1, img2, img3):
        if random.random() < 0.5:
            img1 = tvF.hflip(img1)
            img2 = tvF.hflip(img2)
            img3 = tvF.hflip(img3)
        if random.random() < 0.5:
            rot = random.uniform(-10, 10)
            crop_ratio = rot_crop(rot)
            img1 = tvF.rotate(img1, rot, resample=Image.BILINEAR)
            img2 = tvF.rotate(img2, rot, resample=Image.BILINEAR)
            img3 = tvF.rotate(img3, rot, resample=Image.BILINEAR)
            img1 = tvF.center_crop(img1, int(img1.size[0] * crop_ratio))
            img2 = tvF.center_crop(img2, int(img2.size[0] * crop_ratio))
            img3 = tvF.center_crop(img3, int(img3.size[0] * crop_ratio))
        
        i, j, h, w = self.get_params(img1, self.scale, self.ratio)
        # return the image with the same transformation
        return (tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation), 
                tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation),
                tvF.resized_crop(img3, i, j, h, w, self.size, self.interpolation))

def get_upscale_dataset(args):
    data_dir_path = Path(args.data_dir)
    upscale_dir_path = data_dir_path / 'upscale'

    batch_size = args.batch_size
    input_size = args.input_size
    
    data_randomize = RandomFRC3(256, scale=(0.9, 1.0), ratio=(0.95, 1.05))

    swap_color_space = [RGB2ColorSpace(args.color_space)] 
    random_jitter = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    data_augmentation = [transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS), 
                        transforms.ToTensor()]

    print('reading tagline')
    data_size = args.data_size
    rgb_train_path = data_dir_path / "rgb_train"
    rgb_test_path = data_dir_path / "benchmark"
    sketch_dir_path_list = ["keras_train", "simpl_train", "xdog_train"]
    sketch_dir_path_list = list(map(lambda x : data_dir_path / x, sketch_dir_path_list))
    pixel_dir_path_list = ["keras_train", "rgb_train", "simpl_train", "xdog_train"]
    pixel_dir_path_list = list(map(lambda x : upscale_dir_path / x, pixel_dir_path_list))
    sketch_train_path = data_dir_path / "keras_test"
    
    train_id_list = [int(fid.stem) for fid in rgb_train_path.iterdir() if int(fid.stem) > 0]
    test_id_list = [int(fid.stem) for fid in rgb_test_path.iterdir() if int(fid.stem) > 0]

    print('making train set...')

    train = SuperpixelDataset(rgb_path=rgb_train_path, pixel_path_list=pixel_dir_path_list,
        sketch_path_list=sketch_dir_path_list, file_id_list=train_id_list, 
        override_len=data_size, 
        crop_transform=data_randomize, 
        sketch_transform=transforms.Compose(random_jitter + data_augmentation),
        rgb_transform=transforms.Compose(data_augmentation + swap_color_space))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.thread)

    print(f'train: read {sketch_dir_path_list[0]}, id_list len={len(train_id_list)}')

    print('making test set...')
    test = SuperpixelDataset(rgb_path=rgb_test_path, pixel_path_list=[upscale_dir_path / 'benchmark'],
        sketch_path_list=[sketch_train_path], file_id_list=test_id_list, 
        override_len=100, 
        crop_transform=None, 
        sketch_transform=transforms.Compose(data_augmentation),
        rgb_transform=transforms.Compose(data_augmentation + swap_color_space))

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread)
    
    print(f'test: read {sketch_train_path}, id_list len={len(test_id_list)}')
   
    return train_loader, test_loader

