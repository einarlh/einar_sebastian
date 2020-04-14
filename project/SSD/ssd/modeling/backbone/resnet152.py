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
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers

extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128],
}

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class Resnet152(nn.Module):
        
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE


        self.resnet = resnet152()
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
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)


        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        features.append(x) #38 38
        x = self.resnet.layer3(x) 
        features.append(x) #19 19
        x = self.resnet.layer4(x) 
        features.append(x) # 10 10

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)
        return features

