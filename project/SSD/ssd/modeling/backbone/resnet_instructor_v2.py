import torch.nn as nn
import torch
from torchvision.models import resnet34
from torchvision.models.resnet import BasicBlock   

import torch.nn.functional as F

def add_resnet_blocks(cfg, i):
    in_channels = i
    layers = []
    print(cfg)
    for k, v in enumerate(cfg):
        if not isinstance(in_channels, str):
            print(k)
            print(in_channels)
            if v is 'SQUARE':
                print("Appending SQUARE Convolution")
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(2, 4), stride=2, padding=1)]
                # flag = not flag
            elif v == 'S2':
                print("Appending S2 Convolution")
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(3, 3), stride=2, padding=0)]
            elif v == 'R3':
                print("Appending R3 Resnet Block")
                layers += [(BasicBlock(in_channels, cfg[k + 1], stride = 2, downsample = nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(3, 3), stride=2, padding=1)))]
        in_channels = v
    return layers

def conv3x3AndRelu(inplanes, outplanes, stride = 1, padding = 1):
    return nn.Conv2d(inplanes, outplanes, kernel_size = (3,3), stride = 1, padding = 1)

feat_map_base = {
    '[300, 300]': ['S2', 256, 'R3', 256, 'S2', 256], #convolution + resnet block
    '[320, 240]': ['SQUARE', 256, 'R3', 256, 'S2', 256], #convolution + resnet block
    '[480, 360]': ['SQUARE', 256, 'R3', 256, 'S2', 256], #convolution + resnet block
    # '[320, 240]': [256, 'S', 128, 256, 'S', 256, 128, 'S', 256, 128],
}

class ResnetInstructorV2(nn.Module):
        
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE


        # self.resnet = resnet152()
        # extras_config = extras_base[str(size)]
        feat_map_config = feat_map_base[str(size)]
        # self.extras = nn.ModuleList(add_extras(extras_config, i=2048, size=size))
        self.feat_map = nn.ModuleList(add_resnet_blocks(feat_map_config, i=512))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.feat_map.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def init_from_pretrain(self):
        print("Initializing resnet_instructor_v2 with pretrained weights")
        self.resnet = resnet34(pretrained = True, progress = True)



    def forward(self, x):
        features = []
        # print("Input Shape:" + str(x.shape))
        x = self.resnet.conv1(x)    
        # print("Shape after: conv1" + str(x.shape))
        x = self.resnet.bn1(x)
        # print("Shape after: bn1" + str(x.shape))
        x = self.resnet.relu(x)
        # print("Shape after: relu" + str(x.shape))
        x = self.resnet.maxpool(x)
        # print("Shape after: maxpool" + str(x.shape))


        x = self.resnet.layer1(x)
        # print("Shape after: layer1" + str(x.shape))
        x = self.resnet.layer2(x)
        # print("Appending, Shape after: layer2" + str(x.shape))
        features.append(x) #38 38
        x = self.resnet.layer3(x)
        # print("Appending, Shape after: layer3" + str(x.shape))
        features.append(x) #19 19
        x = self.resnet.layer4(x) 
        # print("Appending, Shape after: layer4" + str(x.shape))
        features.append(x) # 10 10

        for k, v in enumerate(self.feat_map):
            x = F.relu(v(x), inplace=False)
            # print("Shape after: extra k:" + str(k) + " " + str(x.shape))
            if k in [0, 1, 2]:
                # print("appending")
                features.append(x)
        return features

