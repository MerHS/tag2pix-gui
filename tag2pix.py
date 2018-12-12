from network import Generator, Discriminator
import itertools
import utils, time, os, pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from loader.dataloader import get_train_set, get_test_set, get_classid_dict, ColorSpace2RGB, get_tagname_from_tensor
from model.se_resnet import BottleneckX, SEResNeXt
from model.pretrained import se_resnext_half
from tqdm import tqdm

class tag2pix(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 1
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 1
        self.color_revert = ColorSpace2RGB(args.color_space)
        self.layers = args.layers
        self.start_epoch = 0
        self.weight = args.weight
        self.weight_value = args.weight_value

        if args.load == "":
            self.is_load = False
        else:
            self.is_load = True
        self.load_path = self.result_dir + '/' + self.dataset + '/' + args.load + '/'

        self.result_path = self.result_dir + '/' + self.dataset + '/' + time.strftime("%m%d-%H%M%S", time.localtime()) + '/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        with open(self.result_path + 'arguments.txt', 'w') as f:
            f.write(str(args))
            
        # self.class_num = 10
        # read tag list and change 0 to real value
        
        iv_dict, cv_dict, id_to_name, iv_weight_class, cv_weight_class = get_classid_dict(args.tag_dump, self.weight, self.weight_value)

        if self.weight:
            self.iv_weight_class = iv_weight_class
            self.cv_weight_class = cv_weight_class
            self.weight_class = torch.cat([iv_weight_class, cv_weight_class], 0)

        self.color_variant_class_num = len(cv_dict.keys())
        self.color_invariant_class_num = len(iv_dict.keys())
        self.class_num = self.color_variant_class_num + self.color_invariant_class_num

        self.l1_lambda = args.l1_lambda
        self.guide_beta = args.guide_beta
        self.gp_lambda = args.gp_lambda
        self.adv_lambda = args.adv_lambda
        self.use_dragan = args.dragan

        # load dataset
        self.data_loader = get_train_set(args)
        data = self.data_loader.__iter__().__next__()[0]

        # load test data loader
        self.test_data_loader = get_test_set(args)

        self.test_images = self.test_data_loader.__iter__().__next__()
        # save original_, sketch_ image
        original_, sketch_, iv_tag_, cv_tag_ = self.test_images

        # save original_ image tags
        save_tag_path = os.path.join(self.result_path, "tags.txt")

        get_tagname_from_tensor(id_to_name, iv_tag_, cv_tag_, iv_dict, cv_dict, save_tag_path)

        image_frame_dim = int(np.floor(np.sqrt(self.batch_size)))

        if self.gpu_mode:
            original_ = original_.cpu()

        sketch_ = sketch_.data.numpy().transpose(0, 2, 3, 1)
        original_ = self.color_revert(original_)

        utils.save_images(original_[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], 
                          self.result_path + self.model_name + '_original.png')
        utils.save_images(sketch_[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path + self.model_name + '_sketch.png')

        # networks init
        self.Pretrain_ResNeXT = se_resnext_half(dump_path=args.pretrain_dump, num_classes=self.color_invariant_class_num, input_channels=1)
        for param in self.Pretrain_ResNeXT.parameters():
            param.requires_grad = False

        self.G = Generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size,
                cv_class_num=self.color_variant_class_num, iv_class_num=self.color_invariant_class_num, layers=self.layers)
        self.D = Discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size,
                cv_class_num=self.color_variant_class_num, iv_class_num=self.color_invariant_class_num)
        
        utils.initialize_weights(self.G, init_method=args.init_method)
        utils.initialize_weights(self.D, init_method=args.init_method)
        
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        print("gpu mode?", self.gpu_mode)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        print(torch.cuda.device_count(), "GPUS!")

        if self.gpu_mode:
            self.Pretrain_ResNeXT = nn.DataParallel(self.Pretrain_ResNeXT)
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.Pretrain_ResNeXT.to(self.device)
            self.G.to(self.device)
            self.D.to(self.device)
            self.BCE_loss = nn.BCELoss().to(self.device)
            if self.weight:
                self.BCE_weighted_loss = nn.BCELoss(self.weight_class).to(self.device)
            self.CE_loss = nn.CrossEntropyLoss().to(self.device)
            self.L1Loss = nn.L1Loss().to(self.device)
        else:
            self.BCE_loss = nn.BCELoss()
            if self.weight:
                self.BCE_weighted_loss = nn.BCELoss(self.weight_class)
            self.CE_loss = nn.CrossEntropyLoss()
            self.L1Loss = nn.L1Loss()

    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        image_frame_dim = int(np.floor(np.sqrt(self.batch_size)))

        # test_data_loader
        original_, sketch_, iv_tag_, cv_tag_ = self.test_images

        # iv_tag_ to feature tensor 16 * 16 * 256 by pre-reained Sketch.
        with torch.no_grad():  
            feature_tensor = self.Pretrain_ResNeXT(sketch_) 
        if self.gpu_mode:
            original_, sketch_, iv_tag_, cv_tag_, feature_tensor = original_.to(self.device), sketch_.to(self.device), iv_tag_.to(self.device), cv_tag_.to(self.device), feature_tensor.to(self.device)

        G_f, G_g = self.G(sketch_, feature_tensor, cv_tag_)

        if self.gpu_mode:
            G_f = G_f.cpu()
            G_g = G_g.cpu()
        #     G_f = G_f.cpu().data.numpy().transpose(0, 2, 3, 1)
        #     G_g = G_g.cpu().data.numpy().transpose(0, 2, 3, 1)
        # else:
        #     G_f = G_f.data.numpy().transpose(0, 2, 3, 1)
        #     G_g = G_g.data.numpy().transpose(0, 2, 3, 1)

        # # change -1 ~ 1 to rgb 0 ~ 255 
        # G_f = ((G_f + 1) / 2)*255
        # G_g = ((G_g + 1) / 2)*255

        G_f = self.color_revert(G_f)
        G_g = self.color_revert(G_g)
        
        utils.save_images(G_f[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path + self.model_name + '_epoch%03d_G_f_' % (self.start_epoch + epoch) + '.png')
        utils.save_images(G_g[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path + self.model_name + '_epoch%03d_G_g_' % (self.start_epoch + epoch) + '.png')

    def save(self):
        save_dir = self.result_path

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, 'epoch_%d~%d.txt' % (self.start_epoch + 1, self.start_epoch + self.epoch)), 'w') as f:
            f.write("save")

        torch.save({
            'G' : self.G.state_dict(),
            'D' : self.D.state_dict(),
            'G_optimizer' : self.G_optimizer.state_dict(),
            'D_optimizer' : self.D_optimizer.state_dict(),
            'finish_epoch' : self.start_epoch + self.epoch,
            'result_path' : self.result_path
            }, os.path.join(save_dir, self.model_name + '_All.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

        print("============= save success =============")
        print("epoch from %d to %d" % (self.start_epoch + 1, self.start_epoch + self.epoch))
        print("save result path is %s" % self.result_path)

    def load(self):
        load_dir = os.path.join(self.load_path, self.model_name + '_All.pkl')
        
        checkpoint = torch.load(load_dir)


        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        self.start_epoch = checkpoint['finish_epoch']

        print("============= load success =============")
        print("epoch start from %d to %d" % (self.start_epoch + 1, self.start_epoch + self.epoch))
        print("previous result path is %s" % checkpoint['result_path'])