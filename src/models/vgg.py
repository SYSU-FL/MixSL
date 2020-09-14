import torch
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}




class VggLayer(nn.Module):

    def __init__(self, cfg, init_channel=3, last_flag=False, num_classes=10):
        super(VggLayer, self).__init__()
        self.last_flag = last_flag
        self.features = self._make_layers(init_channel, cfg)
        if self.last_flag:
            self.pool = nn.AvgPool2d(kernel_size=1, stride=1)
            self.classifier = nn.Linear(512, num_classes)



    def forward(self, x):
        out = self.features(x)
        if self.last_flag:
            out = self.pool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
        return out


    def _make_layers(self, init_channel, cfg):
        layers = []
        in_channels = init_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


def get_split_vgg16(num_classes=10):
    node_cfg_0 = [64, 64, 'M', 128, 128, 'M']
    node_cfg_1 = [256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    # way1
    #node_cfg_0 = [64, 64, 'M']
    #node_cfg_1 = [128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    # way2
    #node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
    #node_cfg_1 = [512, 512, 512, 'M', 512, 512, 512, 'M']
    # way3
    #node_cfg_0 = [64, 64, 'M', 128]
    #node_cfg_1 = [ 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
    #node_cfg_1 = [512, 512, 512, 'M']
    layer0 = VggLayer(node_cfg_0)
    layer1 = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2], last_flag=True, num_classes=num_classes)
    return layer0, layer1




class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def get_vgg16(num_classes=10):
    return VGG('VGG16', num_classes)
    
if __name__ == '__main__':
   
    node_cfg_0 = [64, 64, 'M', 128, 128, 'M']
    node_cfg_1 = [256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layer0 = VggLayer(node_cfg_0)
    layer1 = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2], last_flag=True)
    









