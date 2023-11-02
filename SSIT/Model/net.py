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
        for i in range(self.netD_num):
            self.netD_dict["netD_{}".format(i)] = ActConvDown(input_ch=input_ch, domain_num=domain_num,
                                                              base_dim=base_dim, n_layers=self.n_layers)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input: torch.Tensor, c:torch.Tensor):
        # Get patch
        feats = []
        result_cam =[]
        result_patch = []
        for i in range(len(self.netD_dict)):
            netD = self.netD_dict["netD_{}".format(i)]
            h1 = input.clone()
            if i>0:
                h1 = self.downsample(h1)
            netD_outputs = netD(x=h1, c=c)
            
            feats.append(netD_outputs.feats)
            result_cam.append(netD_outputs.cam_out)
            result_patch.append(netD_outputs.out)


        return Munch(feats=feats, cam_logit=result_cam, adv_patch=result_patch)


# Generator ----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, img_size, input_ch=3):
        super().__init__()
        h_img_size, w_img_size = img_size
        base_dim=netG_params["base_dim"]
        enc_dec_num = netG_params["enc_dim_num"]

        self.h_img_resize = h_img_size // (2 ** (enc_dec_num - 1))
        self.w_img_resize = w_img_size // (2 ** (enc_dec_num - 1))

        # Input
        self.input_conv = DownBlock(input_ch=input_ch, output_ch=base_dim, k=7, s=1, p=3, act=None, norm=None)
        # Encoder
        self.enc = nn.ModuleList()
        enc_dim_list = [base_dim]
        crr_dim = base_dim
        for i in range(1, enc_dec_num):
            output_dim = min(crr_dim*2, 256)
            if crr_dim != output_dim:
                # Downsample
                self.enc.append(
                DownBlock(input_ch=crr_dim, output_ch=output_dim, k=3, s=2, p=1, act="lrelu", norm="in")
                )
            else:
                #  Bottom
                self.enc.append(
                UpResBlock(input_ch=crr_dim, output_ch=output_dim, act='lrelu', upsample=False)
                )
            enc_dim_list.append(output_dim)
            crr_dim = output_dim
        # Decoder
        self.dec = nn.ModuleList()
        dec_dim_list = list(reversed(enc_dim_list))
        for i in range(len(dec_dim_list)-1):
            input_dim = dec_dim_list[i]
            output_dim = dec_dim_list[i+1]
            if crr_dim != output_dim:
                # Upsample
                self.dec.append(
                    UpResBlock(input_ch=input_dim, output_ch=output_dim, act='lrelu', upsample=True)
                    )
            else:
                # Bottom
                self.dec.append(
                UpResBlock(input_ch=input_dim, output_ch=output_dim, act='lrelu', upsample=False)
                )
        # Output
        self.out_conv = nn.Sequential(*[
            Conv2dBlock(input_ch=enc_dim_list[0], output_ch=3, kernel_size=7, stride=1, padding=3,
                        norm=None, act='tanh')
            ])


    def forward(self, x, y):
        B = x.size()[0]

        src_content = x.clone()
        src_style = y.clone()

        # Encoder
        h = self.input_conv(x)
        #enc_feats = []
        for idx, m in enumerate(self.enc):
            if m.input_ch != m.output_ch:
                h = m(h)
            else:
                h = m(h, src=src_content, ref=src_style)
        for idx, m in enumerate(self.dec):
            if m.input_ch != m.output_ch:
                h = m(h, src=src_style, ref=src_style)
            else:
                h = m(h, src=src_style, ref=src_style)

        out = self.out_conv(h)

        return out