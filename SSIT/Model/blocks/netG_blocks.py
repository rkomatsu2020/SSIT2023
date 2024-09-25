import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import DeformConv2d

from SSIT.Model.blocks.basic_blocks import *

# Deformable Conv -------------------------------------------------------
class DeformConv2d_for_Style(nn.Module):
    def __init__(self, input_ch: int, output_ch: int,
                 k:int, s:int, p:int):
        super().__init__()
        self.k = k
        self.offset_mask = nn.Conv2d(input_ch, 2*k*k, 
                                     kernel_size=k, stride=s, padding=p, 
                                     padding_mode="reflect")
        self.conv = DeformConv2d(input_ch, output_ch, k, s, p)

    def forward(self, x):
        B = x.shape[0]
        # kernel
        x_offset = self.offset_mask(x)
        o1, o2 = torch.chunk(x_offset, 2, dim=1)
        offset = torch.concat([o1, o2], dim=1)
        # conv
        out = self.conv(x, offset=offset)
        return out
    
# Noise -----------------------------------------------------------------
class GaussianNoise(nn.Module):
    def __init__(self, input_ch, img_size):
        super().__init__()
        self.input_ch = input_ch
        self.noise_scaler = nn.Parameter(torch.randn(1, input_ch, 1, 1))
    def forward(self, x):
        B, C, H, W = x.size()
        # Add mean=0 std=1 Gaussian noise
        noise = torch.randn([B, 1, H, W], device=x.device)
        return x + noise * self.noise_scaler

# Norm -------------------------------------------------------------------
class DirectNorm2d(nn.Module):
    def __init__(self, input_ch: int, style_dim: int, **kargs):
        super().__init__()

        self.eps = 1e-8
        self.input_norm = nn.InstanceNorm2d(input_ch)

        # Conv
        k, s, p = 3, 1, 1
        self.conv = DeformConv2d_for_Style(style_dim, input_ch, k, s, p)


    def forward(self, input, style):
        B, C, H, W = input.size()
        input = self.input_norm(input)

        s = self.conv(style)
        style_mean, style_var = torch.mean(s, dim=[2,3], keepdim=True), torch.var(s, dim=[2,3], keepdim=True)
        style_std = torch.sqrt(style_var + self.eps)

        out = style_std * input + style_mean

        return out

# Encoder -------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int, hidden_ch: int=None,
                 downsample: bool = False):
        super().__init__()
        self.downsample = downsample
        self.input_ch = input_ch
        self.output_ch = output_ch
        hidden_ch = input_ch if hidden_ch is None else hidden_ch

        if self.downsample is True:
            k, s, p = 3, 2, 1
        else:
            k, s, p = 3, 1, 1

        # Conv
        self.conv0 = Conv2dBlock(input_ch=input_ch, output_ch=output_ch,
                                kernel_size=k, stride=s, padding=p,
                                norm="in", act="prelu")

    def forward(self, x):
        h = x.clone() 

        h = self.conv0(h)
        return h
    

# Decoder -------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, img_size: tuple, input_ch: int, output_ch: int, style_emb: int, hidden_ch: int=None,
                 upsample: bool=False, shortcut: bool=True, **kargs):
        super().__init__()
        self.src_imgsize = img_size
        self.upsample = upsample
        self.input_ch = input_ch
        self.output_ch = output_ch
        
        self.shortcut = shortcut
        hidden_ch = output_ch if hidden_ch is None else hidden_ch

        H1, W1 = img_size
        if self.upsample is True:
            H2 = H1*2
            W2 = W1*2
        else:
            H2, W2 = img_size
        k, s, p = 3, 1, 1

        # Layer1: C -> B -> R 
        if self.upsample is True:
            self.conv0 =  nn.Conv2d(input_ch//4, hidden_ch, 
                                    k, s, p, padding_mode="reflect")
        else:
            self.conv0 =  nn.Conv2d(input_ch, hidden_ch, 
                                    k, s, p, padding_mode="reflect")

        self.noise0 = GaussianNoise(hidden_ch, img_size=(H1, W1))
        self.norm0 = DirectNorm2d(hidden_ch, style_dim=style_emb)
        self.act0 = nn.PReLU()

        # Layer2: C -> B -> R 
        self.conv1 =  nn.Conv2d(hidden_ch, output_ch, 
                                k, s, p, padding_mode="reflect")
        self.noise1 = GaussianNoise(hidden_ch, img_size=(H2, W2))
        self.norm1 = DirectNorm2d(hidden_ch, style_dim=style_emb)
        self.act1 = nn.PReLU()

        # Upsample
        if self.upsample is True:
            self.up = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x, ref):
        h = x.clone()

        if self.upsample is True:
            h = self.up(h)

        h = self.conv0(h)
        h = self.noise0(h)
        h = self.norm0(h, style=ref)
        h = self.act0(h)

        h = self.conv1(h)
        h = self.noise1(h)
        h = self.norm1(h, style=ref)
        h = self.act1(h)

        if self.shortcut is True:
            return h + x
        else:
            return h
        
