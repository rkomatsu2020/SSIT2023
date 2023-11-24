import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops.layers.torch import Rearrange

from SSIT.Model.blocks.basic_blocks import *

# Noise -----------------------------------------------------------------
class GaussianNoise(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        B = x.size()[0]
        # Add mean=0 std=1 Gaussian noise
        noise = torch.randn(x.size()).to(x.device) * self.weight.view(1, -1, 1, 1)
        return x + noise

# Norm -------------------------------------------------------------------
class DirectNorm2d(nn.Module):
    def __init__(self, num_features: int, img_size:tuple, style_dim:int =256):
        super().__init__()
        self.eps = 1e-8
        self.input_norm = nn.InstanceNorm2d(num_features)
        self.fc = nn.Sequential(*[
                Conv2dBlock(3*3, num_features, 3, 1, 1, act='lrelu', norm=None)
        ])

    def forward(self, input, style):
        B, C, H, W = input.size()
        input = self.input_norm(input)


        style = self.fc(style)
        style_mean, style_var = torch.mean(style, dim=[2,3], keepdim=True), torch.var(style, dim=[2,3], keepdim=True)
        style_std = torch.sqrt(style_var + self.eps)
        out = style_std * input + style_mean

        return out

# Encoder -------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int, k: int, s:int, p:int,
                 act: str='lrelu', norm: str="in"):
        super().__init__()

        self.input_ch = input_ch
        self.output_ch = output_ch

        self.noise = GaussianNoise(input_ch)
        self.conv = Conv2dBlock(input_ch=input_ch, output_ch=output_ch, kernel_size=k, stride=s, padding=p,
                                norm=norm, act=act, act_inplace=False, act_first=True)
        self.act = get_act(act)
        self.norm = get_norm(norm, output_ch)

        
    def forward(self, x):
        x = self.noise(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)
        x = self.conv(x)
        
        return x

# Decoder -------------------------------------------------
class UpResBlock(nn.Module):
    def __init__(self, img_size: tuple, input_ch: int, output_ch: int, hidden_ch: int=None,
                 act: str='lrelu', pad_type: str='reflect', upsample: bool=False):
        super().__init__()
        self.upsample = upsample
        self.input_ch = input_ch
        self.output_ch = output_ch
        
        self.learned_shortcut = (input_ch != output_ch)
        hidden_ch = input_ch if hidden_ch is None else hidden_ch

        self.noise0 = GaussianNoise(input_ch)
        self.noise1 = GaussianNoise(hidden_ch)
        # CNN
        self.conv0 =  Conv2dBlock(input_ch, hidden_ch, 3, 1, 1, pad_type='reflect', norm=None, act=None)
        self.conv1 = Conv2dBlock(hidden_ch, output_ch, 3, 1, 1, pad_type='reflect', norm=None, act=None)

        if self.learned_shortcut is True:
            self.conv_s = Conv2dBlock(input_ch, output_ch, 1, 1, 0, pad_type='reflect', norm=None, act=None)

        # BatchNorm
        self.norm0 = DirectNorm2d(input_ch, img_size=img_size)
        self.norm1 = DirectNorm2d(hidden_ch, img_size=img_size)
        # Act
        self.act0 = get_act(act)
        self.act1 = get_act(act)
        # Upsample
        if self.upsample is True:
            self.up = nn.Upsample(scale_factor=2)

    def shortcut(self, x):
        if self.upsample is True:
            x = self.up(x)

        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x

        return x_s

    def forward(self, x, ref):
        x_s = self.shortcut(x)

        h = self.noise0(x)
        h = self.norm0(h, style=ref)
        h = self.act0(h)
        h = self.conv0(h)
        

        if self.upsample is True:
            h = self.up(h)

        h = self.noise1(h)
        h = self.norm1(h, style=ref)
        h = self.act1(h)
        h = self.conv1(h)
        

        out = h + x_s

        return out