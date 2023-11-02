from munch import Munch
import torch
import torch.nn.functional as F
import torch.nn as nn

from SSIT.Model.blocks.basic_blocks import *


class ActConvDown(nn.Module):
    def __init__(self, input_ch: int, domain_num:int, base_dim: int, n_layers: int):
        super().__init__()
        self.netD = nn.ModuleDict()
        self.n_layers = n_layers
        # Input Conv
        self.netD["conv_0"] = Conv2dBlock(input_ch=input_ch, output_ch=base_dim, kernel_size=3, stride=1, padding=1,
                                        norm=None, act=None, act_inplace=False, use_spectral=False)
        # Hidden Conv
        crr_dim = base_dim
        for j in range(1, self.n_layers):
            output_dim = min(crr_dim*2, 256)
            #self.netD["norm_{}".format(j)] = DirectNorm2d(num_features=crr_dim)
            self.netD["conv_{}".format(j)] = ActFirstResBlock(input_ch=crr_dim, output_ch=output_dim, 
                                                              act="lrelu", act_inplace=False, downsample=True, use_spectral=True)
            
            
            crr_dim = output_dim
        # Output CAM logit
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(crr_dim, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(crr_dim, 1, bias=False))
        self.conv1x1 = Conv2dBlock(input_ch=crr_dim*2, output_ch=crr_dim, kernel_size=1, stride=1, act="lrelu", use_spectral=False, act_first=True, act_inplace=False)
        self.cam_lambda = nn.Parameter(torch.zeros(1))
        # Output conv
        self.netD["out"] = nn.Sequential(*[nn.LeakyReLU(0.2, False), nn.Conv2d(crr_dim, domain_num, 4, 1, 0)])

    def get_cam_logit(self, x):
        x0 = x.clone()
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # Output feat
        x = torch.cat([gap, gmp], 1)
        out = self.conv1x1(x) * self.cam_lambda + x0
        # Output CAM
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)

        return cam_logit, out

    def forward(self, x, c):
        feats = list()
        h = x.clone()

        for j in range(self.n_layers):
            h = self.netD["conv_{}".format(j)](h)
            if j != 0:
                # Append feats
                feats.append(h.clone())

        # Append CAM
        cam_out, h = self.get_cam_logit(h)
        # Append patch
        out = self.netD["out"](h)
        # Select by c
        idx = torch.LongTensor(range(c.size(0))).to(x.device)
        out = out[idx, c]     

        return Munch(feats=feats, cam_out=cam_out, 
                     out=out)
