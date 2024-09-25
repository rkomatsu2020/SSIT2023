import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from SSIT.config import model_setting

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
                #rnd = 0.9 + (1 - 0.9) * torch.rand_like(input.data.cpu())
                rnd = 0.9
                real_tensor = torch.ones(input.size()) * rnd
                if torch.cuda.is_available():
                    real_tensor = real_tensor.to(self.device)
            target_tensor = Variable(real_tensor, requires_grad=False)
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                #rnd = 0 + (0.1 - 0) * torch.rand_like(input.data.cpu())
                rnd = 0.1
                fake_tensor = torch.ones(input.size()) * rnd
                if torch.cuda.is_available():
                    fake_tensor = fake_tensor.to(self.device)
            target_tensor = Variable(fake_tensor, requires_grad=False)
        
        return target_tensor

    def __call__(self, inputs, is_real):
        loss = 0
        if isinstance(inputs, list):
            for i in inputs:
                target_tensor = self.get_target_tensor(input=i, is_real=is_real)
                loss += self.loss(i, target_tensor)

        else:
            target_tensor = self.get_target_tensor(input=inputs, is_real=is_real)
            loss = self.loss(inputs, target_tensor)

        return loss
    
def tv_loss(img, tv_weight):
    B, C, H, W = img.size()
    w_var = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_var = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_var + w_var) / (B * C * H * W)
    return loss
    
class ViTLoss(nn.Module):
    def __init__(self, vit:nn.Module):
        super().__init__()
        self.vit = vit
        self.content_loss = nn.L1Loss()
        self.style_loss = nn.MSELoss()
        '''ViT memo:
        self.ViT = VitExtractor(device=device)
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        '''

    def calc_content_loss(self, outputs, targets):
        def attn_cosine_sim(x, eps=1e-08):

            norm1 = x.norm(dim=2, keepdim=True)
            factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
            sim_matrix = (x @ x.permute(0, 2, 1)) / factor
            return sim_matrix

        #layers = [i for i in range(11+1)]
        layers = [6]
        weight = [2**(-(len(layers)-1-i)) for i in range(len(layers))]

        B = outputs.shape[0]
        outputs = self.vit.vit_input_img_normalization(outputs)
        targets = self.vit.vit_input_img_normalization(targets)
        outputs = self.vit.get_features(outputs, layers)
        targets = self.vit.get_features(targets, layers)

        loss = 0
        for idx,(x,y) in enumerate(zip(outputs, targets)):
            x = attn_cosine_sim(x)
            y = attn_cosine_sim(y)
            loss += self.content_loss(x, y.detach()) * weight[idx]
            
        return loss
    
    def calc_style_loss(self, outputs, targets):
        '''
        outputs: translated images
        targets: ref images
        '''
        layers = [11]
        weight = [1]
        B = outputs.shape[0]
        outputs = self.vit.vit_input_img_normalization(outputs)
        targets = self.vit.vit_input_img_normalization(targets)
        outputs = self.vit.get_features(outputs, layers)
        targets = self.vit.get_features(targets, layers)

        loss = 0
        for idx,(x,y) in enumerate(zip(outputs, targets)):
            x = x[:, 0, :]
            y = y[:, 0, :]
            loss += self.style_loss(x, y.detach()) * weight[idx]

        return loss