from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops.layers.torch import Rearrange

from Utils import unnormalize
from SSIT.Model.blocks.basic_blocks import Conv2dBlock
from SSIT.config import netG_params, netD_params
from SSIT.Model.blocks.netG_blocks import DownBlock, UpBlock
from SSIT.Model.blocks.netD_blocks import ViTDiscriminator

# Discriminator -------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_ch=3, **kargs):
        super().__init__()
        self.net = ViTDiscriminator(input_ch=input_ch, 
                                    img_size=kargs["img_size"], domain_num=kargs["domain_num"])

        self.apply(init_weights)

    def forward(self, input: torch.Tensor, **kargs):
        out = self.net(input, domain=kargs["domain"])
        return Munch(adv_patch=out["patch"])


# Generator ----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, img_size:tuple, input_ch=3):
        super().__init__()
        enc_dim_list = netG_params["enc_dim"]
        bottom_dim_list = netG_params["bottom_dim"]
        dec_dim_list = netG_params["dec_dim"]
        
        # Input
        self.input_conv = Conv2dBlock(input_ch, enc_dim_list[0], 
                                      kernel_size=7, stride=1, padding=3, pad_type="reflect",
                                      norm=None, act=None)
        # Encoder
        self.enc = nn.ModuleList()
        self.enc_img_size = [img_size]
        self.down_count = 0
        for i in range(1, len(enc_dim_list)):
            input_dim = enc_dim_list[i-1]
            output_dim = enc_dim_list[i]
            H, W = self.enc_img_size[-1]

            self.enc.append(
                DownBlock(input_ch=input_dim, output_ch=output_dim, 
                          downsample=True)
            )
            self.enc_img_size.append((H//2, W//2))
        # Style
        self.style_emb = 3 * 2

        # Bottom
        self.bottom = nn.ModuleList()
        for i in range(len(bottom_dim_list)-1):
            input_dim = bottom_dim_list[i]
            output_dim = bottom_dim_list[i+1]
            img_size = self.enc_img_size[-1]

            self.bottom.append(
                UpBlock(input_ch=input_dim, output_ch=output_dim, style_emb=self.style_emb, 
                        upsample=False, img_size=img_size, shortcut=True,
                        )
            )

        # Decoder
        self.dec = nn.ModuleList()
        for i in range(len(dec_dim_list)-1):
            input_dim = dec_dim_list[i]
            output_dim = dec_dim_list[i+1]
            img_size = self.enc_img_size[len(dec_dim_list)-1-i]

            self.dec.append(
                UpBlock(input_ch=input_dim, output_ch=output_dim, style_emb=self.style_emb, 
                            upsample=True, img_size=img_size, shortcut=False,
                            )
            )

        # Output
        self.out_conv = nn.Sequential(*[
            Conv2dBlock(input_ch=dec_dim_list[-1], output_ch=3, 
                        kernel_size=7, stride=1, padding=3, pad_type="reflect",
                        norm=None, act='tanh', use_bias=False)
            ])
        
        self.apply(init_weights)

    def style_extractor(self, x, H, W):
        y = F.adaptive_avg_pool2d(x, (H, W))
        z = F.adaptive_max_pool2d(x, (H, W))
        return torch.cat([y, z], dim=1)

    def forward(self, x, y):
        src_H, src_W = y.shape[2], y.shape[3]
        # Encoder
        enc_list = list()
        h = self.input_conv(x)
        for idx, m in enumerate(self.enc):
            h = m(h)
            enc_list.append(h.clone())
        # Bottom
        for idx, m in enumerate(self.bottom):
            s = self.style_extractor(y, H=h.shape[2], W=h.shape[3])
            h = m(h, ref=s)
        # Decoder
        for idx, m in enumerate(self.dec):
            s = self.style_extractor(y, H=h.shape[2]*2, W=h.shape[3]*2)
            h = m(h, ref=s)

        out = self.out_conv(h)

        return out

# Weights init --------------------------------------
def init_weights(m, mode="kaiming"):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if mode == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif mode == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)