import torch.nn as nn
import torch
from torchvision.models import googlenet
import torch.nn.functional as F
import numpy as np


def add_extras(cfg, i, size=300):
    # Extra layers added to Resnet for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'SQUARE' and in_channels != 'SMALLER' and in_channels != 'F':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            if v == 'SMALLER':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(3, 3), stride=2, padding=1)]
            elif v is 'SQUARE':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(2, 4), stride=2, padding=1)]
                # flag = not flag
            elif v == 'F':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(2,2), stride=1, padding=0)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], stride = 1)]
            flag = not flag
        in_channels = v
    # if size == 512:
    #     layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
    #     layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers

extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128],
    '[320, 240]': [256, 'SQUARE', 512, 256, 'SMALLER', 512, 512, 'SMALLER', 512, 512, 'F', 512, 1024],
}

class GoogleNet(nn.Module):
        
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE


        # self.googlenet = googlenet()
        extras_config = extras_base[str(size)]
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self):
        print("Initializing googlenet with pretrained weights")
        self.googlenet = googlenet(pretrained = True, progress = True)

    def forward(self, x):
        features = []
        # print("Input Shape:" + str(x.shape))
        x = self.googlenet.conv1(x)    
        #print("Shape after: conv1 " + str(x.shape))
        x = self.googlenet.maxpool1(x)
        #print("Shape after: maxpool1 " + str(x.shape))
        x = self.googlenet.conv2(x)
        #print("Shape after: conv2 " + str(x.shape))
        x = self.googlenet.conv3(x)
        #print("Shape after: conv3 " + str(x.shape))
        x = self.googlenet.maxpool2(x)
        #print("Shape after: maxpool2 " + str(x.shape))
        x = self.googlenet.inception3a(x)
        #print("Shape after: inception3a " + str(x.shape))
        x = self.googlenet.inception3b(x)
        # print("Shape after: inception3b " + str(x.shape))
        # print("Appending")
        features.append(x)
        x = self.googlenet.maxpool3(x)
        #print("Shape after: maxpool3 " + str(x.shape))
        x = self.googlenet.inception4a(x)
        #print("Shape after: inception4a " + str(x.shape))
        x = self.googlenet.inception4b(x)
        #print("Shape after: inception4b " + str(x.shape))
        x = self.googlenet.inception4c(x)
        #print("Shape after: inception4c " + str(x.shape))
        x = self.googlenet.inception4d(x)
        #print("Shape after: inception4d " + str(x.shape))
        x = self.googlenet.inception4e(x)
        # print("Shape after: inception4e " + str(x.shape))
        # print("Appending")
        features.append(x)
        x = self.googlenet.maxpool4(x)
        #print("Shape after: maxpool4 " + str(x.shape))
        x = self.googlenet.inception5a(x)
        #print("Shape after: inception5a " + str(x.shape))
        x = self.googlenet.inception5b(x)
        # print("Shape after: inception5b " + str(x.shape))
        # print("Appending")
        features.append(x)        

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # print("Shape after: extra k:" + str(k) + " " + str(x.shape))
            if k in [2, 4, 8]:
                # print("appending")
                features.append(x)
        return features

