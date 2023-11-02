import torch
import torch.nn as nn
import kornia
from torch.autograd import Variable

class GANLoss(nn.Module):
    def __init__(self, loss, device):
        super(GANLoss,self).__init__()
        self.device = device
        self.real_label_var = None
        self.fake_label_var = None
        
        self.loss = loss

    def get_target_tensor(self, input, is_real):
        target_tensor = None
        if is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                #rnd = 0.9 + (1 - 0.9) * torch.rand_like(input.data.cpu())
                real_tensor = torch.ones(input.size()) * 0.9
                if torch.cuda.is_available():
                    real_tensor = real_tensor.to(self.device)
            target_tensor = Variable(real_tensor, requires_grad=False)
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                #rnd = 0 + (0.1 - 0) * torch.rand_like(input.data.cpu())
                fake_tensor = torch.zeros(input.size())
                if torch.cuda.is_available():
                    fake_tensor = fake_tensor.to(self.device)
            target_tensor = Variable(fake_tensor, requires_grad=False)
        
        return target_tensor

    def __call__(self, inputs, is_real):
        loss = 0
        if isinstance(inputs, list):
            for input in inputs:
                target_tensor = self.get_target_tensor(input, is_real)
                loss += self.loss(input, target_tensor)
        else:
            target_tensor = self.get_target_tensor(inputs, is_real)
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
        self.style_criterion = nn.L1Loss()
        self.weights = [1/32, 1/16, 1/8, 1/4, 1.0]
        #self.weights = [1/4, 1.0]

    def calc_style_loss(self, x, y):
        def gram_matrix(input):
            b, ch, h, w = input.size()
            features = input.view(b, ch, w * h)
            features_t = features.transpose(1, 2)
            gram = torch.matmul(features, features_t)
            return gram
        
        x_vgg, y_vgg = self.vgg(x, mode='style'), self.vgg(y, mode='style')

        loss = 0
        for i in range(len(x_vgg)):
            gram_x = gram_matrix(x_vgg[i])
            gram_y = gram_matrix(y_vgg[i]).detach()
            loss += self.style_criterion(gram_x, gram_y)

        return loss

    def calc_content_loss(self, x, y):
        B, C, H, W = x.size()
        x = vgg_normalization(x)
        y = vgg_normalization(y)
        #x = kornia.color.rgb_to_grayscale(x).expand(B, 3, H, W)
        #y = kornia.color.rgb_to_grayscale(y).expand(B, 3, H, W)
        
        x_vgg, y_vgg = self.vgg(x, mode='content'), self.vgg(y, mode='content')
        
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.content_criterion(x_vgg[i], y_vgg[i].detach())
        return loss


    def forward(self, x, y, mode='content'):
        if mode == 'content':
            return self.calc_content_loss(x, y)
        elif mode == 'style':
            return self.calc_style_loss(x, y)

def gradient_penalty(y, x):
    weight = torch.ones(y.size()).to(y.device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

class StyleMeanStdLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_mean, x_std = self.calc_mean_std(x)
        y_mean, y_std = self.calc_mean_std(y)

        mean_loss = self.criterion(x_mean, y_mean)
        std_loss = self.criterion(x_std, y_std)

        return mean_loss + std_loss

    def calc_mean_std(self, x, eps=1e-8):
        B, C, H, W = x.size()
        x_mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)

        x_var = x.view(B, C, -1).var(dim=2) + eps
        x_std = torch.sqrt(x_var).view(B, C, 1, 1)

        return x_mean, x_std