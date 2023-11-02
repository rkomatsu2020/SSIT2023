import torch
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
import torch.nn as nn

def get_act(act, inplace=True):
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'lrelu':
        return nn.LeakyReLU(0.2, inplace)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == "gelu":
        return nn.GELU()
    else:
        return None

def get_norm(norm, norm_ch):
    if norm == 'bn':
        return nn.BatchNorm2d(norm_ch)
    elif norm == 'in':
        return nn.InstanceNorm2d(norm_ch)
    elif norm == 'layer':
        return nn.GroupNorm(1, norm_ch)
    elif norm == 'pixel':
        return PixelNorm()
    else:
        return None

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

class ResBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int, hidden_ch: int=None, norm: str=None, use_spectral: bool=False, 
                 act: str='lrelu', act_inplace: bool=True, downsample: bool=False):
        super().__init__()
        self.learned_shortcut = (input_ch != output_ch)
        self.downsample = downsample

        hidden_ch = input_ch if hidden_ch is None else hidden_ch
        
        self.conv0 = Conv2dBlock(input_ch, hidden_ch, 3, 1, 1, pad_type='reflect', 
                                    norm=norm, act=act, act_inplace=act_inplace, use_spectral=use_spectral)
        self.conv1 = Conv2dBlock(hidden_ch, output_ch, 3, 1, 1, pad_type='reflect', 
                                    norm=norm, act=act, act_inplace=act_inplace, use_spectral=use_spectral)
        if self.learned_shortcut is True:
            self.conv_s = Conv2dBlock(input_ch, output_ch, 1, 1, 0, pad_type='reflect',
                                      norm=norm, act=None, use_spectral=use_spectral)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x.clone()

        dx = self.conv0(x)
        if self.downsample is True:
            dx = F.avg_pool2d(dx, 2)
            x_s = F.avg_pool2d(x_s, 2)
        dx = self.conv1(dx)
        out = x_s + dx
        return out

class ActFirstResBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int, hidden_ch: int=None, norm: str=None, 
                 use_spectral: bool=False, act: str='lrelu', act_inplace: bool=False, downsample: bool=False):
        super().__init__()
        self.learned_shortcut = (input_ch != output_ch)
        self.downsample = downsample

        hidden_ch = input_ch if hidden_ch is None else hidden_ch
        
        k = 3
        s = 1
        self.conv0 = Conv2dBlock(input_ch, hidden_ch, k, s, 1, pad_type='reflect', 
                                    norm=norm, act=act, act_first=True, act_inplace=act_inplace, use_spectral=use_spectral)
        self.conv1 = Conv2dBlock(hidden_ch, output_ch, k, 1, 1, pad_type='reflect', 
                                    norm=norm, act=act, act_first=True, act_inplace=act_inplace, use_spectral=use_spectral)
        if self.learned_shortcut is True:
            self.conv_s = Conv2dBlock(input_ch, output_ch, 3, 1, 1, pad_type='reflect',
                                      norm=norm, act=None, use_spectral=use_spectral)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x.clone()
        dx = self.conv0(x)
        if self.downsample is True:
            dx = F.avg_pool2d(dx, 2)
            x_s = F.avg_pool2d(x_s, 2)
        dx = self.conv1(dx)
        out = x_s + dx
        return out

class LinearBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm=None, act='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_ch, output_ch, bias=use_bias)

        # initialize normalization
        norm_ch = output_ch
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_ch)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_ch)
        else:
            self.norm = None

        # initialize activation
        self.act = get_act(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, padding=0, n_groups=1,
                 norm=None, act='relu', act_inplace=True, pad_type='reflect', use_spectral=False, use_bias=True, act_first=False):
        super().__init__()
        self.use_bias = use_bias
        self.act_first = act_first
        # initiaize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        # initialize normalization
        norm_ch = output_ch
        self.norm = get_norm(norm, norm_ch)
        # initialize activation
        self.act = get_act(act, act_inplace)
        # initialize conv
        if use_spectral is False:
            self.conv = nn.Conv2d(input_ch, output_ch, kernel_size, stride, bias=self.use_bias, groups=n_groups)
        else:
            self.conv = spectral_norm(nn.Conv2d(input_ch, output_ch, kernel_size, stride, bias=self.use_bias, groups=n_groups))
    def forward(self, x, **kargs):
        if self.act_first is True:
            # Act
            if self.act is not None:
                x = self.act(x)
            # Conv
            x = self.conv(self.pad(x))
            # Norm
            if self.norm is not None:
                x = self.norm(x)
        else:
            # Conv
            x = self.conv(self.pad(x))
            # Norm
            if self.norm is not None:
                x = self.norm(x)
            # Act
            if self.act is not None:
                x = self.act(x)

        return x
