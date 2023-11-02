import torch.nn as nn
import torchvision

# VGG19 ----------------------------------------------------------
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize
        self.content = [1, 6, 11, 20, 29]
        # Load pretrained VGG
        # memo -------------------------------------
        # conv1_1:0, conv1_2:2, conv2_1:5, conv2_2:7 conv3_1:10, conv3_2:12, conv3_3:14, conv3_4:16
        # conv4_1:19, conv4_2:21, conv4_3:23, conv4_4:25, conv5_1:28, conv5_2:30, conv5_3:32, conv5_4:34
        # ------------------------------------------

        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        vgg_len = len(vgg_pretrained_features)
        self.extract_vgg = nn.ModuleList()
        for x in range(vgg_len):
            self.extract_vgg.append(vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x, mode='content'):
        # Set target layer
        target = self.content
        # Extract map
        out = []
        for idx, layer in enumerate(self.extract_vgg):
            x = layer(x)
            if idx in target:
                out.append(x.clone())

        return out