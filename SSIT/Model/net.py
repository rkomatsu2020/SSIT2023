from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F

from SSIT.Model.blocks.basic_blocks import Conv2dBlock
from SSIT.config import netG_params, netD_params
from SSIT.Model.blocks.netG_blocks import DownBlock, UpResBlock
from SSIT.Model.blocks.netD_blocks import ActConvDown


# Discriminator -------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_size, input_ch=3, domain_num=2):
        super().__init__()
        self.img_size = img_size
        base_dim=netD_params["base_dim"]
        self.netD_num = netD_params["netD_num"]
        self.n_layers = netD_params["n_layers"]

        self.netD_dict = nn.ModuleDict()
        for i in range(1, self.netD_num+1):
            self.netD_dict["netD_{}".format(i)] = ActConvDown(input_ch=input_ch, domain_num=domain_num,
                                                              base_dim=base_dim, n_layers=self.n_layers)
        netD_output_dim = self.netD_dict["netD_{}".format(self.netD_num)].output_dim
        self.domain_fc = nn.Linear(netD_output_dim, domain_num)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input: torch.Tensor):
        B = input.size(0)
        # Get patch
        feats = []
        result_cam =[]
        result_patch = []
        for i in range(1, len(self.netD_dict)+1):
            netD = self.netD_dict["netD_{}".format(i)]
            h1 = input.clone()
            if i>1:
                h1 = self.downsample(h1)
            netD_outputs = netD(x=h1)
            
            feats.append(netD_outputs.feats)
            result_cam.append(netD_outputs.cam_out)
            result_patch.append(netD_outputs.out)
        # Get class
        h = feats[-1][-1]
        h = F.adaptive_avg_pool2d(h, 1)
        out_cls = self.domain_fc(h.view(B, -1))

        return Munch(feats=feats, cam_logit=result_cam, adv_patch=result_patch, pred_class=out_cls)


# Generator ----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, img_size, input_ch=3, patch_size: int=32):
        super().__init__()
        self.h, self.w = img_size
        h_img_size, w_img_size = img_size

        base_dim=netG_params["base_dim"]
        max_dim = netG_params["base_dim"]*8
        enc_dec_num = netG_params["enc_dim_num"]

        self.h_img_resize = h_img_size
        self.w_img_resize = w_img_size

        # Input
        self.input_conv = DownBlock(input_ch=input_ch, output_ch=base_dim, k=7, s=1, p=3, act=None, norm=None)
        # Encoder
        self.enc = nn.ModuleList()
        enc_dim_list = [base_dim]
        crr_dim = base_dim
        for i in range(1, enc_dec_num):
            output_dim = min(crr_dim*2, max_dim)
            if crr_dim != output_dim:
                self.enc.append(
                    DownBlock(input_ch=crr_dim, output_ch=output_dim, k=3, s=2, p=1,
                                act="lrelu", norm="in")
                )
                self.h_img_resize = self.h_img_resize // 2
                self.w_img_resize = self.w_img_resize // 2
            else:
                self.enc.append(
                    DownBlock(input_ch=crr_dim, output_ch=output_dim, k=3, s=1, p=1,
                                act="lrelu", norm="in")
                )
            enc_dim_list.append(output_dim)
            crr_dim = output_dim

        style_dim = enc_dim_list[-1]
        self.style_pooling = nn.ModuleDict(
            {
                'pool_1': nn.AvgPool2d(kernel_size=(3, 3)),
                'pool_2': nn.AvgPool2d(kernel_size=(5, 5)),
                'pool_3': nn.AvgPool2d(kernel_size=(7, 7)),

                'style_rho': nn.Linear(3*3, 2)
            }
        )
        self.eps = 1e-8

        # Decoder
        self.dec = nn.ModuleList()
        dec_dim_list = list(reversed(enc_dim_list))
        for i in range(len(dec_dim_list)-1):
            input_dim = dec_dim_list[i]
            output_dim = dec_dim_list[i+1]

            if crr_dim != output_dim:
                self.dec.append(
                    UpResBlock(
                            img_size=img_size,
                            input_ch=input_dim, output_ch=output_dim, 
                            act='lrelu', upsample=True)
                    )
            else:
                self.dec.append(
                UpResBlock(
                    img_size=img_size,
                    input_ch=input_dim, output_ch=output_dim, 
                    act='lrelu', upsample=False)
                )

        # Output
        self.out_conv = nn.Sequential(*[
            Conv2dBlock(input_ch=enc_dim_list[0], output_ch=3, kernel_size=7, stride=1, padding=3,
                        norm=None, act='tanh')
            ])
        
    def style_pool(self, x):
        B, C, _, _ = x.size()

        y = None
        for i in range(1, 3+1):
            z = self.style_pooling["pool_{}".format(i)](x)
            z = F.interpolate(z, size=(self.h_img_resize, self.w_img_resize), mode='bilinear', align_corners=True)
            
            if i == 1:
                y = z.clone()
            else:
                y = torch.cat([y, z.clone()], dim=1)

        c = F.adaptive_avg_pool2d(y, 1)
        c = c.view(B, -1)
        softmax = nn.Softmax(1)
        rho = softmax(self.style_pooling["style_rho"](c))
        # Norm
        in_mean, in_var = torch.mean(y, dim=[2, 3], keepdim=True), torch.var(y, dim=[2, 3], keepdim=True)
        input_in = (y - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(y, dim=[1, 2, 3], keepdim=True), torch.var(y, dim=[1, 2, 3], keepdim=True)
        input_ln = (y - ln_mean) / torch.sqrt(ln_var + self.eps)

        out = rho[:, 0].view(B, 1, 1, 1) * input_in + rho[:, 1].view(B, 1, 1, 1) * input_ln

        return out

    def forward(self, x, y):
        src_content = x.clone()
        src_style = y.clone()

        # Encoder
        h = self.input_conv(x)
        s = self.style_pool(y)
        #enc_feats = []
        for idx, m in enumerate(self.enc):
            h = m(h)

        for idx, m in enumerate(self.dec):
            h = m(h, ref=s)

        out = self.out_conv(h)

        return out

