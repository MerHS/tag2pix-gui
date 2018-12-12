import argparse, os, torch
from GAN import GAN
from CGAN import CGAN
from LSGAN import LSGAN
from DRAGAN import DRAGAN
from ACGAN import ACGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from infoGAN import infoGAN
from EBGAN import EBGAN
from BEGAN import BEGAN
from tag2pix import tag2pix

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loader', 'tag_dump.pkl')
PRETRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth')

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='tag2pix',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'tag2pix'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='tag2pix', choices=['tag2pix', 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                        help='The name of dataset')

    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=256, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    
    parser.add_argument('--pretrain_dump', default=PRETRAIN_PATH)
    parser.add_argument('--data_dir', default=DATA_DIRECTORY)
    parser.add_argument('--tag_dump', default=TAG_FILE_PATH)
    parser.add_argument('--data_size', default=60000, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--thread', type=int, default=8)

    parser.add_argument('--l1_lambda', type=float, default=1000)
    parser.add_argument('--guide_beta', type=float, default=0.9)
    parser.add_argument('--layers', type=int, nargs='+', default=[12,8,5,5])

    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--color_space', type=str, default='rgb', choices=['lab', 'rgb', 'hsv'],
                        help='Learning color space')

    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--init_method', type=str, default='kaiming', choices=['kaiming', 'normal'])
    parser.add_argument('--dragan', action='store_true')
    parser.add_argument('--gp_lambda', type=float, default=0.5)
    parser.add_argument('--adv_lambda', type=float, default=1)
    parser.add_argument('--load', type=str, default="")
    parser.add_argument('--weight', type=bool, default=False)
    parser.add_argument('--weight_value', type=int, nargs='+', default=[10,10])

    print(parser.parse_args())
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

        # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'CGAN':
        gan = CGAN(args)
    elif args.gan_type == 'ACGAN':
        gan = ACGAN(args)
    elif args.gan_type == 'infoGAN':
        gan = infoGAN(args, SUPERVISED=False)
    elif args.gan_type == 'EBGAN':
        gan = EBGAN(args)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args)
    elif args.gan_type == 'WGAN_GP':
        gan = WGAN_GP(args)
    elif args.gan_type == 'DRAGAN':
        gan = DRAGAN(args)
    elif args.gan_type == 'LSGAN':
        gan = LSGAN(args)
    elif args.gan_type == 'BEGAN':
        gan = BEGAN(args)
    elif args.gan_type == 'tag2pix':
        gan = tag2pix(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
