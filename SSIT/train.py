import sys
import pathlib
sys.path.append(pathlib.Path(__file__).resolve().parents[0].as_posix())
sys.path.append(pathlib.Path(__file__).resolve().parents[1].as_posix())
sys.path.append(pathlib.Path(__file__).resolve().parents[2].as_posix())

import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from Utils.DataParser.get_data_loader import get_train_loader, InputFetcher
from Utils.DataParser.set_dataset_path import set_dataset_path
from Utils.ModelParser.model_parser import set_and_get_save_dir, save_models, load_model

from SSIT.Model.net import Generator, Discriminator
from SSIT.Model.vgg import VGG19
from SSIT.Model.loss import *
from SSIT.config import model_setting
from test_Guided_Trans import Test


class TrainSSIT():
    def __init__(self, gpu_no=0, n_epoch=100, batch_size=1, 
                 dataset_name='photo2art', **kargs):
        # Environment
        self.base_dir = os.path.dirname(__file__)
        self.model_name = model_setting['model_name']
        self.gpu_no = gpu_no
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.device = 'cuda:{}'.format(gpu_no) if torch.cuda.is_available() else 'cpu'
        # Get dataset
        img_size_config = 'default' if dataset_name not in model_setting['img_size'].keys() else dataset_name
        img_size = model_setting['img_size'][img_size_config]

        dataset = set_dataset_path(dataset_name, is_train=True)
        source_loader = get_train_loader(root=dataset['source'], which='source', img_size=img_size,
                                         batch_size=self.batch_size, target_domain_dic=dataset['target_label_dic'], 
                                         k_shot=None, pad_crop=model_setting['pad_crop'])
        ref_loader = get_train_loader(root=dataset['ref'], which='reference', img_size=img_size,
                                      batch_size=self.batch_size, target_domain_dic=dataset['target_label_dic'], 
                                      k_shot=model_setting['k_shot'], pad_crop=model_setting['pad_crop'])
        self.train_loader = InputFetcher(loader=source_loader, loader_ref=ref_loader, 
                                         mode='train', device=self.device)
        self.iter_per_epoch = min(len(source_loader), len(ref_loader))
        self.total_iter = self.n_epoch * self.iter_per_epoch
        self.domain_num = dataset['style_num']
        # Loss Parameter
        adv_weight = 'default' if dataset_name not in model_setting['lambda_adv'].keys() else dataset_name
        cyc_weight = 'default' if dataset_name not in model_setting['lambda_cyc'].keys() else dataset_name
        style_weight = 'default' if dataset_name not in model_setting['lambda_style'].keys() else dataset_name

        self.lambda_adv = float(model_setting['lambda_adv'][adv_weight])
        self.lambda_cyc = float(model_setting['lambda_cyc'][cyc_weight])
        self.lambda_style = float(model_setting['lambda_style'][style_weight])
        self.unrolled_steps = model_setting['unroll_steps']
        # About save model path
        self.trained_models_dir_path = set_and_get_save_dir(self.model_name,
                                                        '@trained_with_{}(c={} s={} a={})'.format(dataset_name, self.lambda_cyc, self.lambda_style, self.lambda_adv)
                                                        )
        # Set model & path
        self.netG = Generator(img_size=img_size, input_ch=3).to(self.device)
        self.netD = Discriminator(img_size=img_size, input_ch=3, domain_num=self.domain_num).to(self.device)
        self.netG.apply(self.init_weights)
        self.netD.apply(self.init_weights)
        # Set Optimizers
        lr = 0.0002
        self.g_lr = lr/2
        self.d_lr = lr*2
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                             lr=self.g_lr, betas=(0.0, 0.9))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=self.d_lr, betas=(0.0, 0.9))
        self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer_G, milestones=[self.total_iter//2, self.total_iter//4*3], gamma=0.1)
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer_D, milestones=[self.total_iter//2, self.total_iter//4*3], gamma=0.1)
        # Set loss
        self.VGG_Loss = VGGLoss(self.device, VGG19())
        self.Adv_Loss = GANLoss(loss=nn.MSELoss(), device=self.device, domain_num=self.domain_num)
        self.Style_Loss = nn.L1Loss()
        self.Cls_Loss = nn.CrossEntropyLoss()
        # Set test
        self.test = Test(gpu_no=gpu_no, dataset_name=dataset_name, model_config=model_setting)

    @staticmethod
    def get_trainable_params(dataset_name: str, style_domain_num: int, content_domain_num:int):
        from torchinfo import summary
        import contextlib
        from thop import profile

        def get_MB_from_params(params):
            params_B = params * 4
            params_kB = params_B/ 1024
            params_MB = params_kB / 1024
            return params_MB

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_size_config = 'default' if dataset_name not in model_setting['img_size'] else dataset_name
        img_size = model_setting['img_size'][img_size_config]
        target_domain_num = style_domain_num
        rnd_input_size = (1, 3, img_size[0], img_size[1])
        input1 = torch.randn(rnd_input_size).to(device)
        input2 = torch.randn(rnd_input_size).to(device)

        netG = Generator(img_size=img_size, input_ch=3).to(device)
        netD = Discriminator(img_size=img_size, input_ch=3, domain_num=target_domain_num).to(device)
        # Get Flops & params
        netG_flops, netG_params = profile(netG, inputs=(input1, input2))
        netG_params = get_MB_from_params(netG_params)
        # Print params (netG)
        netG_param_txt = "{}-params(netG) for {}.txt".format(model_setting["model_name"], dataset_name)
        with open(netG_param_txt, "a") as f:
            with contextlib.redirect_stdout(f):
                print("Generator 1 ----------------------------------------------------")
                summary(netG, input_size=[rnd_input_size, rnd_input_size], dtypes=[torch.float, torch.float])
                print("\n")
                print("================================================================")

            f.write("Flops: {}\n".format(netG_flops))
            f.write("Total Model Params(MB): {}\n".format(netG_params))

        # Print params (netD)
        netD_flops, netD_params = profile(netD, inputs=(input1, ))
        netD_params = get_MB_from_params(netD_params)
        netD_param_txt = "{}-params(netD) for {}.txt".format(model_setting["model_name"], dataset_name)
        with open(netD_param_txt, "a") as f:
            with contextlib.redirect_stdout(f):
                print("Discriminator 1 ------------------------------------------------")
                summary(netD, input_size=[rnd_input_size], dtypes=[torch.float])
                print("\n")
                print("================================================================")

            f.write("Flops: {}\n".format(netD_flops))
            f.write("Total Model Params(MB): {}\n".format(netD_params))


    def set_train_mode(self):
        model_list = [self.netG, self.netD]
        for model in model_list:
            model.train()

    def reload_resume(self, num):
        self.netG = load_model(save_dir=self.trained_models_dir_path, model=self.netG, model_name='netG', num=num)
        self.netD = load_model(save_dir=self.trained_models_dir_path, model=self.netD, model_name='netD', num=num)

    def __call__(self, save_interval=1, resume_num: int =None):
        # Re-Load trained model
        if resume_num is None:
            start_iter = 0
        else:
            self.reload_resume(resume_num)
            start_iter = self.iter_per_epoch * resume_num
        # Conditional Translation Train ----------------------------
        for iter_num in range(start_iter, self.total_iter):
            self.set_train_mode()
            inputs = next(self.train_loader)

            A = Variable(inputs.x_src.to(self.device)) # Content Img & Domain
            B = Variable(inputs.x_ref.to(self.device))
            srcDomain = inputs.y_src.to(self.device)
            trgDomain = inputs.y_ref.to(self.device)

            epoch = iter_num//self.iter_per_epoch+1
            iter = (iter_num+1) % self.iter_per_epoch if (iter_num+1) % self.iter_per_epoch != 0 else self.iter_per_epoch
            print('------------------------------------------------------------------------------------------------------')
            print('epoch:{} {}/{} train with {} for {}'.format(epoch, iter, self.iter_per_epoch, self.model_name, self.dataset_name))

            self.update(As=(A, srcDomain), Bs=(B, trgDomain), 
                        iter_num=iter, unrolled_steps=self.unrolled_steps)
            print('------------------------------------------------------------------------------------------------------')

            self.scheduler_G.step()
            self.scheduler_D.step()

            if epoch % save_interval == 0 and iter == self.iter_per_epoch:
                save_models(
                    self.trained_models_dir_path, 
                    model_list=[self.netG, self.netD],
                    model_name_list = ['netG', 'netD'],
                    num = epoch,
                    device = self.device
                    )
                self.test(epoch)

        save_models(
                self.trained_models_dir_path, 
                model_list=[self.netG, self.netD],
                model_name_list = ['netG', 'netD'],
                num = self.n_epoch,
                device = self.device
                    )

    def update(self, As: tuple, Bs: tuple, 
               iter_num: int, unrolled_steps: int=None,
               **kargs):
        if unrolled_steps is None:
            self.update_D(As, Bs) 
            self.update_G(As, Bs) 
        else:
            if iter_num % self.unrolled_steps == 0:
                self.update_D(As, Bs)
            self.update_G(As, Bs)

    def update_G(self, As: tuple, Bs: tuple, **kargs): 
        A, A_domain = As
        B, B_domain = Bs
        lossG = {}
        self.optimizer_G.zero_grad()
        # Generate fake img
        A2B = self.netG(A, B)
        # Discriminate
        pred_fake = self.netD(input=A2B)
        lossG['Adv_Fake'] = self.Adv_Loss(pred_fake.adv_patch, True) * self.lambda_adv
        lossG['Adv_Fake(CAM)'] = self.Adv_Loss(pred_fake.cam_logit, True) * self.lambda_adv
        lossG['Adv_Fake(Cls)'] = self.Cls_Loss(pred_fake.pred_class, B_domain)

        pred_real = self.netD(input=B)
        sum_feat_loss = 0
        for idx, (fake_feats, real_feats) in enumerate(zip(pred_fake.feats, pred_real.feats)):
            for fake, real in zip(fake_feats, real_feats):
                feat_loss = self.Style_Loss(fake, real.detach())
                sum_feat_loss += feat_loss
        lossG['Style_Feat'] = sum_feat_loss * self.lambda_style

        lossG['Content_Feat(VGG)'] = self.VGG_Loss(A2B, A, 'content') * self.lambda_cyc
        
        loss = sum(lossG.values())
        loss.backward()
        self.optimizer_G.step()

        print('\tGenerator')
        for k, v in lossG.items():
            print('\t\t{}:{:.5f}'.format(k, v.item()))


    def update_D(self, As: tuple, Bs: tuple, 
                 output_log=True, **kargs): 
        A, A_domain = As
        B, B_domain = Bs
        
        lossD = {}
        B.requires_grad_()
        self.optimizer_D.zero_grad()
        with torch.no_grad():
            A2B = self.netG(A, B)
        # Discriminate
        pred_real = self.netD(input=B)
        pred_fake = self.netD(input=A2B.detach())

        lossD['Adv_Real'] = self.Adv_Loss(pred_real.adv_patch, True) * self.lambda_adv
        lossD['Adv_Fake'] = self.Adv_Loss(pred_fake.adv_patch, False) * self.lambda_adv
        lossD['Adv_Real(CAM)'] = self.Adv_Loss(pred_real.cam_logit, True) * self.lambda_adv
        lossD['Adv_Fake(CAM)'] = self.Adv_Loss(pred_fake.cam_logit, False) * self.lambda_adv
        lossD['Adv_Real(Cls)'] = self.Cls_Loss(pred_real.pred_class, B_domain)

        loss = sum(lossD.values())
        loss.backward()
        self.optimizer_D.step()

        if output_log is True:
            print('\tDiscriminator')
            for k, v in lossD.items():
                print('\t\t{}:{:.5f}'.format(k, v.item()))

    def init_weights(self, m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)