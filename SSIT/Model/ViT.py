import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

'''
Ref from: https://github.com/omerbt/Splice/blob/master/models/extractor.py
    from: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
'''

# ViT ------------------------------------------------------------
def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    #dino_vitb8 # ['dino_vitb8', 'dino_vits8', 'dino_vitb16', 'dino_vits16']
    def __init__(self, device, model_name='dino_vits16', pretrained: bool=True):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model_name = model_name

        if pretrained is True:
            # Freeze params
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
    
    def get_features(self, x, layers:list):
        x = self.model.prepare_tokens(x)

        output = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in layers:
                output.append(self.model.norm(x))
        return output
    
    def vit_input_img_normalization(self, input_img, minmax_norm:bool=True):
        '''
        Normalizing for fitting ImageNet dataset
        '''
        input_img = F.interpolate(input_img, (224, 224))
        device = input_img.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

        if minmax_norm is True:
            min_vals = torch.amin(input_img, dim=(2, 3), keepdim=True)
            max_vals = torch.amax(input_img, dim=(2, 3), keepdim=True)
            input_img = (input_img - min_vals) / (max_vals - min_vals) # -> to [0, 1]
            
        return (input_img - mean) / std

# ----------------------------------------------------------------

