import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision

from Utils import mkdir, unnormalize
from Utils.DataParser.get_data_loader import get_test_loader, InputFetcher
from Utils.DataParser.set_dataset_path import set_dataset_path
from Utils.ModelParser.model_parser import set_and_get_save_dir, load_model

from SSIT.Model.net import Generator
from SSIT.config import model_setting

class Test():
    def __init__(self, gpu_no, dataset_name, model_config):
        self.base_dir = os.path.dirname(__file__)
        self.device = 'cuda:{}'.format(gpu_no) if torch.cuda.is_available() else 'cpu'
        self.model_name = model_config['model_name']
        self.dataset_name = dataset_name
        self.img_size = model_config['img_size']['default'] if dataset_name not in model_config['img_size'].keys() else model_config['img_size'][dataset_name]
        # About trained model path
        self.trained_model_dir = set_and_get_save_dir(self.model_name, dataset_name)
        # Init loader & model
        self.has_loader = False
        self.loaded_trained_model = False
        # Set Generator
        img_size_config = 'default' if dataset_name not in model_setting['img_size'] else dataset_name
        img_size = model_setting['img_size'][img_size_config]
        self.netG = Generator(img_size=img_size, input_ch=3).to(self.device)

    def init_loader(self):
        # Sample Num
        self.sample_num = 5
        # Set test data loader
        dataset = set_dataset_path(self.dataset_name, is_train=False)
        source_loader, self.src_transform = get_test_loader(root=dataset['source'], which='source', batch_size=self.sample_num, target_domain_dic=None, img_size=self.img_size)
        ref_loader, _ = get_test_loader(root=dataset['ref'], which='reference', batch_size=self.sample_num, img_size=self.img_size)
        self.test_loader = InputFetcher(loader=source_loader, loader_ref=ref_loader,
                                        mode='test', device=self.device)
        # Get domain name info
        self.target_domain_dic = dataset['target_label_dic']
        self.target_name_dic = dataset['target_name_dic']
        self.content_domain_num = dataset['content_num']
        self.target_domain_num = len(self.target_domain_dic)


    def imshow(self, img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), save_img_path=None, plt_imshow=True):
        npimg = img.data.cpu()
        npimg = unnormalize(npimg, mean, std)
        if save_img_path is not None:
            torchvision.utils.save_image(torch.Tensor(npimg), save_img_path)
        # Image showing on plt
        if plt_imshow is True:
            plt.imshow(np.transpose(npimg, (1,2,0)))
        # Return output img
        else:
            output_img = np.transpose(npimg, (1,2,0)) * 255
            return np.array(output_img, np.uint8)

    def __call__(self, num):
        # Get loader
        if self.has_loader is False:
            self.init_loader()
            self.has_loader = True
        # Create result path
        result_dir = '{}/results'.format(self.trained_model_dir)
        mkdir(result_dir)
        # Load trained model
        self.netG = load_model(save_dir=self.trained_model_dir, model=self.netG, model_name='netG', num=num)
        self.netG.eval()
        # Load test img
        data = next(self.test_loader)
        content_img = data.x_src
        style_img = data.x_ref
        style_domain = data.y_ref

        fig = plt.figure()
        for idx_i, imgs in enumerate(zip(content_img, style_img)):
            c, s = imgs
            c = c.to(self.device)
            c = c.unsqueeze_(0)
            s = s.to(self.device)
            s = s.unsqueeze_(0)
            domain = style_domain[idx_i]
            # show content
            plt.subplot(self.sample_num, 3, 1 + idx_i*3)
            plt.xticks([], [])
            plt.yticks([], [])
            self.imshow(c)
            # show style
            plt.subplot(self.sample_num, 3, 2 + idx_i*3)
            plt.xticks([], [])
            plt.yticks([], [])
            self.imshow(s)
            # show translated result
            with torch.no_grad():
                fake_img = self.netG(c, s)
            plt.subplot(self.sample_num, 3, 3 + idx_i*3)
            plt.xticks([], [])
            plt.yticks([], [])
            self.imshow(fake_img)
            

        plt.savefig('{}/epoch{}_fakeStyle_result.png'.format(result_dir, num), bbox_inches="tight", pad_inches=0.05)
        plt.close()

    def save_translated_img(self, num, content_path, style_path):
        # Get loader
        if self.has_loader is False:
            self.init_loader()
            self.has_loader = True

        if self.loaded_trained_model is False:
            # Load trained model
            self.netG = load_model(save_dir=self.trained_model_dir, model=self.netG, model_name='netG', num=num)
            self.netG.eval()
            self.loaded_trained_model = True

        # Get img
        content_input = Image.open(content_path)
        style_input = Image.open(style_path)
        content_name = os.path.basename(content_path)
        content_name = content_name.split(".")[0]
        style_name = os.path.basename(style_path)
        style_name = style_name.split(".")[0]
        # Create result path
        result_dir = '{}/results'.format(self.trained_model_dir)
        if os.path.exists(result_dir) is False:
            os.mkdir(result_dir)

        if self.loaded_trained_model is False:
            # Load trained model
            self.netG = load_model(save_dir=self.trained_model_dir, model=self.netG, model_name='netG', num=num)
            self.netG.eval()
            self.loaded_trained_model = True

        if isinstance(content_input, torch.Tensor) is False:
            c = self.src_transform(content_input)
            s = self.src_transform(style_input)
        else:
            c = content_input.clone()
            s = content_input.clone()

        c = c.to(self.device)
        s = s.to(self.device)
        if len(c.size()) != 4:
            c = c.unsqueeze_(0)
            s = s.unsqueeze_(0)
        # show translated result
        with torch.no_grad():
            fake_img = self.netG(c, s)

        fig = plt.figure()
        plt.xticks([], [])
        plt.yticks([], [])
        self.imshow(fake_img)

        plt.savefig('{}/epoch{}_c={}-s={}_result.png'.format(result_dir, num, content_name, style_name), bbox_inches="tight", pad_inches=0.05)
        plt.close()


    def get_translated_img(self, num, content_input, style_input, get_tensor_output=False):
        if self.loaded_trained_model is False:
            # Load trained model
            self.netG = load_model(save_dir=self.trained_model_dir, model=self.netG, model_name='netG', num=num)
            self.netG.eval()
            self.loaded_trained_model = True

        if isinstance(content_input, torch.Tensor) is False:
            # Get loader
            if self.has_loader is False:
                self.init_loader()
                self.has_loader = True
            # Transform
            c = self.src_transform(content_input)
            s = self.src_transform(style_input)
        else:
            c = content_input.clone()
            s = content_input.clone()

        c = c.to(self.device)
        s = s.to(self.device)
        if len(c.size()) != 4:
            c = c.unsqueeze_(0)
            s = s.unsqueeze_(0)
        # show translated result
        with torch.no_grad():
            fake_img = self.netG(c, s)

        if get_tensor_output is False:
            fake_img = self.imshow(fake_img, plt_imshow=False)

            return fake_img

        else:
            return fake_img.data.cpu()
        
