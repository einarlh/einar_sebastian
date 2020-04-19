import torch.nn as nn
import torch
from torchvision.models import resnet152
import torch.nn.functional as F


def add_extras(cfg, i, size=300):
    # Extra layers added to Resnet for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S' and in_channels != 'SQUARE':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v == 'SQUARE':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(2, 4), stride=2, padding=1)]
                flag = not flag
            elif v == 'F':
                layers += [nn.Conv2d(in_channels, cfg[k - 1], kernel_size=(2,2), stride=1, padding=0)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # if size == 512:
    #     layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
    #     layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers

extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128],
    '[320, 240]': [256, 'SQUARE', 512, 128, 'S', 256, 256],
}

class Resnet152(nn.Module):
        
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE


        # self.resnet = resnet152()
        extras_config = extras_base[str(size)]
        self.extras = nn.ModuleList(add_extras(extras_config, i=2048, size=size))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self):
        print("Initializing resnet152 with pretrained weights")
        self.resnet = resnet152(pretrained = True, progress = True)

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

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # print("Shape after: extra k:" + str(k) + " " + str(x.shape))
            if k in [1, 3, 4]:
                # print("appending")
                features.append(x)
        return features

