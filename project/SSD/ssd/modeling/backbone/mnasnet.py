import torch.nn as nn
import torch
from torchvision.models import mnasnet1_0
import torch.nn.functional as F


def add_extras(cfg, i):
    # Extra layers added to Resnet for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S' and in_channels != 'K':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v == 'K':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(2,3), stride=2, padding=0)]
                flag = not flag
            elif v == 'F':
                layers += [nn.Conv2d(in_channels, cfg[k - 1], kernel_size=(2,2), stride=1, padding=0)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128],
    '[320, 240]': [320, 'S', 512, 128, 'S', 256, 128, 'K', 256],
}

class MNASNet(nn.Module):
        
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE


        self.mnasnet = mnasnet1_0()
        extras_config = extras_base[str(size)]
        self.extras = nn.ModuleList(add_extras(extras_config, i=320))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self):
        # print("Initializing mnasnet with pretrained weights")
        self.mnasnet = mnasnet1_0(pretrained = True, progress = True)

    def forward(self, x):
        features = []
        # print("Input Shape:" + str(x.shape))
        for i in range(10):
            x = self.mnasnet.layers[i](x)    
            # print("Shape at: i:" + str(i) + " " + str(x.shape))
        features.append(x)
        # print("Appending")
        for i in range(10, 12):
            x = self.mnasnet.layers[i](x)    
            # print("Shape at: i:" + str(i) + " " + str(x.shape))
        features.append(x)
        # print("Appending")
        for i in range(12, 14):
            x = self.mnasnet.layers[i](x)   
            # print("Shape at: i:" + str(i) + " " + str(x.shape))
        features.append(x)
        # print("Appending")

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # print("Shape after: extra k:" + str(k) + " " + str(x.shape))
            if k % 2 == 1:
                # print("appending")
                features.append(x)
        return features

