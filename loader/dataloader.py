import cv2
import pickle
import torch
import numpy as np
from pathlib import Path
from random import uniform
from PIL import Image
from skimage import color
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets
from .xdog_blend import blend_xdog_and_sketch, add_intensity

TAG_SIMPLE_BACKGROUND = 412368
TAG_WHITE_BACKGROUND = 515193
TAG_SOLO = 212816

def pseudo_uniform(id, a, b):
    return (((id * 1.253 + a * 324.2351 + b * 534.342) * 20147.2312369804) + 0.12949) % (b - a) + a

def real_uniform(id, a, b):
    return uniform(a, b)

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
            if is_tag[0].item() == 1.0:
                tag_name = id_to_name[iv_dict_inverse[i]]
                tag_list.append(tag_name)
                f.write(tag_name + " ")
        iv_tag_name_list.append(tag_list)
        f.write("\n")

    f.write("cv tags\n")
    
    for batch_unit in cv_tensor:
        tag_list = [] 
        for i, is_tag in enumerate(batch_unit):
            if is_tag[0].item() == 1.0:
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
            iv_weight_class = torch.zeros(vector.size())

        min_cv = torch.min(cv_weight_class)
        range_cv = torch.max(cv_weight_class) - min_cv

        if range_cv > 0:
            cv_weight_class = weight_value[1] * (cv_weight_class - min_cv) / range_cv
        else:
            cv_weight_class = torch.zeros(vector.size())

        print("iv loss weight is %d, cv loss weight is %d" % (weight_value[0], weight_value[1]))
        # print(iv_weight_class, cv_weight_class)
        return (iv_dict, cv_dict, id_to_name, iv_weight_class, cv_weight_class)
    else:
        return (iv_dict, cv_dict, id_to_name, None, None)


def read_tagline_txt(tag_txt_path, img_dir_path, iv_dict, iv_class_len, cv_dict, cv_class_len, data_size=0, simple=False, is_train=True):
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
    all_tag_num = 0
    iv_tag_num = 0
    cv_tag_num = 0

    with tag_txt_path.open('r') as f:
        for line in f:
            tag_list = list(map(int, line.split(' ')))
            file_id = tag_list[0]
            tag_list = tag_list[1:]

            if not (img_dir_path / f'{file_id}.png').exists():
                continue

            if len(tag_list) < 8:
                continue
            
            iv_class = torch.zeros(iv_class_len, dtype=torch.float)
            cv_class = torch.zeros(cv_class_len, dtype=torch.float)
            

            if simple:
                if not (TAG_SOLO in tag_list):
                    continue
                    
                if not (TAG_SIMPLE_BACKGROUND in tag_list or TAG_WHITE_BACKGROUND in tag_list):
                    continue


            tag_exist = False
            # iv_tag_num = 0
            # iv_tag_name_list = []

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
            # cv_tag_num = 0
            # cv_tag_name_list = []

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

    print(count, all_tag_num, iv_tag_num, cv_tag_num)
    return (file_id_list, iv_class_list, cv_class_list)


class ColorAndSketchDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, iv_class_list, cv_class_list,
            override_len=None, sketch_transform=None, color_transform=None, is_train=True, **kwargs):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.iv_class_list = iv_class_list
        self.cv_class_list = cv_class_list
        self.sketch_transform = sketch_transform
        self.data_len = len(file_id_list)
        self.is_train = is_train
        self.rand_gen = real_uniform if is_train else pseudo_uniform
        self.override_len = override_len
        self.color_transform = color_transform

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        iv_tag_class = self.iv_class_list[index]
        cv_tag_class = self.cv_class_list[index]

        color_path = self.image_dir_path / f"{file_id}.png"
        sketch_path = self.image_dir_path / f"{file_id}_sk.png"

        color_img = Image.open(color_path).convert('RGB') 

        blend_img = self._sketch_blend(file_id, color_path, sketch_path)
        sketch_img = Image.fromarray(blend_img) # to [1, H, W]

        if self.color_transform is not None:
            color_img = self.color_transform(color_img)
        if self.sketch_transform is not None:
            sketch_img = self.sketch_transform(sketch_img)

        return (color_img, sketch_img, iv_tag_class, cv_tag_class)

    def __len__(self):
        if self.override_len > 0 and self.data_len > self.override_len:
            return self.override_len
        return self.data_len

    def _sketch_blend(self, file_id, illust_path, sketch_path):
        blend = self.rand_gen(file_id, -0.5, 0.25)
        rand_gen = self.rand_gen
        sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)

        if blend > 0:
            illust = cv2.imread(str(illust_path))
            intensity = rand_gen(file_id, 1., 2.)
            degamma = rand_gen(file_id, 1./intensity, 1.)
            k = rand_gen(file_id, 2.0, 3.0)
            sigma = rand_gen(file_id, 0.35, 0.45)
            blend_img = blend_xdog_and_sketch(illust, sketch, intensity=intensity, degamma=degamma, k=k, sigma=sigma)
        else:
            intensity = rand_gen(file_id, 1., 1.3)
            blend_img = add_intensity(sketch, intensity)

        return blend_img

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

def make_dataloader(args, data_dir_path, is_train=False, shuffle=True):
    batch_size = args.batch_size
    input_size = args.input_size
    
    sketch_normalize = [transforms.Normalize(mean=[0.9184], std=[0.1477]), ]
    color_normalize = [RGB2ColorSpace(args.color_space), ] # [lambda img: (img * 2 - 1.), ]
    data_augmentation = [transforms.Resize((input_size, input_size)), 
                        transforms.ToTensor()]
    
    iv_dict, cv_dict, id_to_name, iv_weight_class, cv_weight_class = get_classid_dict(args.tag_dump, args.weight, args.weight_value)
    iv_class_len = len(iv_dict.keys())
    cv_class_len = len(cv_dict.keys())

    print('reading tagline')
    data_size = args.data_size if is_train else args.data_size // 10
    tag_path = data_dir_path / "tags.txt"
    (train_id_list, train_iv_class_list, train_cv_class_list) = read_tagline_txt(tag_path, data_dir_path, 
        iv_dict, iv_class_len, cv_dict, cv_class_len, data_size=data_size, simple=args.simple, is_train=is_train)

    print('making dataset...')
    train = ColorAndSketchDataset(data_dir_path, train_id_list, iv_class_list=train_iv_class_list, cv_class_list=train_cv_class_list,
        override_len=data_size, sketch_transform=transforms.Compose(data_augmentation + sketch_normalize),
        color_transform=transforms.Compose(data_augmentation + color_normalize), is_train=is_train)

    print('making dataloader...')
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, num_workers=args.thread)
    
    print(f'iv_class_len={iv_class_len}, cv_class_len={cv_class_len}')
    print(f'read {data_dir_path}, id_list len={len(train_id_list)}, iv_class len={len(train_iv_class_list)}, cv_class len={len(train_cv_class_list)}')
   
    return train_loader

def get_train_set(args):
    data_dir = args.data_dir
    train_dir_path = Path(data_dir) / ("train" if not args.valid else "validation")
    return make_dataloader(args, train_dir_path, is_train=True)

def get_test_set(args):
    data_dir = args.data_dir
    test_dir_path = Path(data_dir) / "test"
    return make_dataloader(args, test_dir_path, is_train=False, shuffle = False)


def dataloader(args, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    if args.dataset == 'tag2pix':
        if args.test:
            data_loader = get_test_set(args)
        else:
            data_loader = get_train_set(args)

    elif args.dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader
