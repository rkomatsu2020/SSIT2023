import torch
import torch.nn as nn
from torch.autograd import Variable

def onehot_encode(trgDomain, domain_num, scalar=None):
    eye = torch.eye(domain_num)
    eye = eye[trgDomain].view(-1, domain_num)
    if scalar is None:
        return eye
    else:
        return eye * scalar
    
def onehot_tile(is_real, input, trgDomain, domain_num: int):
    B, C, H, W = input.size()
    if is_real is True:
        rnd_scaler = 0.9 + (1 - 0.9) * torch.rand((B, domain_num))
        code = onehot_encode(trgDomain=trgDomain.data.cpu(), domain_num=domain_num) * rnd_scaler
    else:
        rnd_scaler = (1 - 0.9) * torch.rand((B, domain_num))
        code = torch.ones((B, domain_num)) * rnd_scaler

    code = code.view(B, domain_num, 1, 1)
    out = code.expand((B, C, H, W))

    return out.to(input.device)



class GANLoss(nn.Module):
    def __init__(self, domain_num, loss, device):
        super(GANLoss,self).__init__()
        self.domain_num = domain_num

        self.device = device
        self.real_label_var = None
        self.fake_label_var = None
        
        self.loss = loss

    def get_target_tensor(self, input, is_real):
        target_tensor = None
        if is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                rnd = 0.9 + (1 - 0.9) * torch.rand_like(input.data.cpu())
                real_tensor = torch.ones(input.size()) * rnd
                if torch.cuda.is_available():
                    real_tensor = real_tensor.to(self.device)
            target_tensor = Variable(real_tensor, requires_grad=False)
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                rnd = 0 + (0.1 - 0) * torch.rand_like(input.data.cpu())
                fake_tensor = torch.ones(input.size()) * rnd
                if torch.cuda.is_available():
                    fake_tensor = fake_tensor.to(self.device)
            target_tensor = Variable(fake_tensor, requires_grad=False)
        
        return target_tensor

    def __call__(self, inputs, is_real, target_idx=None):
        loss = 0
        if isinstance(inputs, list):
            for i in inputs:
                if target_idx is None:
                    target_tensor = self.get_target_tensor(input=i, is_real=is_real)
                else:
                    target_tensor = onehot_tile(input=i, is_real=is_real, trgDomain=target_idx, domain_num=self.domain_num)
                
                loss += self.loss(i, target_tensor)

        else:
            if target_idx is None:
                target_tensor = self.get_target_tensor(input=inputs, is_real=is_real)
            else:
                target_tensor = onehot_tile(input=inputs, is_real=is_real, trgDomain=target_idx, domain_num=self.domain_num)
            loss = self.loss(inputs, target_tensor)

        return loss

def vgg_normalization(img):
    img = (img + 1) / 2 # convert [-1, 1] to [0, 1] (mean=-1, std=2)
    device = img.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    return (img - mean) / std

class VGGLoss(nn.Module):
    def __init__(self, device:str, vgg:nn.Module):
        super(VGGLoss, self).__init__()
        self.vgg = vgg.to(device)
        self.content_criterion = nn.L1Loss()

    def calc_content_loss(self, x, y):
        B, C, H, W = x.size()
        x = vgg_normalization(x)
        y = vgg_normalization(y)
        
        x_vgg, y_vgg = self.vgg(x, mode='content'), self.vgg(y, mode='content')
        
        loss = 0
        for i in range(len(x_vgg)):
            weight = 1 / pow(2, (len(x_vgg)-1-i))
            loss += weight * self.content_criterion(x_vgg[i], y_vgg[i].detach())
        return loss


    def forward(self, x, y, mode='content'):
        if mode == 'content':
            return self.calc_content_loss(x, y)
        elif mode == 'style':
            return self.calc_style_loss(x, y)