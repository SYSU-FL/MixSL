import torch
from torch import nn as nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
import time




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResInputLayer(nn.Module):
    def __init__(self):
        super(ResInputLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class ResBlockLayer(nn.Module):
    def __init__(self, block, planes, num_blocks, stride, in_planes=None):
        super(ResBlockLayer, self).__init__()
        if in_planes is not None:
            self.in_planes = in_planes
        else:
            self.in_planes = 64
        self.layer = self._make_layer(block, planes, num_blocks, stride)

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out

    def get_in_plances(self):
        return self.in_planes


class ResOutputLayer(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResOutputLayer, self).__init__()
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



"the base"

class ResNet18_Extractor(nn.Module):
    def __init__(self):
        super(ResNet18_Extractor, self).__init__()
        self.layer0 = ResInputLayer()
        self.layer1 = ResBlockLayer(BasicBlock, 64, 2, 1, 64)
        self.layer2 = ResBlockLayer(BasicBlock, 128, 2, 2, 64)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class ResNet18_Classifer(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_Classifer, self).__init__()
        self.layer0 = ResBlockLayer(BasicBlock, 256, 2, 2, 128)
        self.layer1 = ResBlockLayer(BasicBlock, 512, 2, 2, 256)
        self.layer2 = ResOutputLayer(BasicBlock, num_classes)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out



"larger extractor"
# class ResNet18_Extractor(nn.Module):
#     def __init__(self):
#         super(ResNet18_Extractor, self).__init__()
#         self.layer0 = ResInputLayer()
#         self.layer1 = ResBlockLayer(BasicBlock, 64, 2, 1, 64)
#         self.layer2 = ResBlockLayer(BasicBlock, 128, 2, 2, 64)
#         self.layer3 = ResBlockLayer(BasicBlock, 256, 2, 2, 128)
#     def forward(self, x):
#         out = self.layer0(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         return out


# class ResNet18_Classifer(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18_Classifer, self).__init__()
#         self.layer1 = ResBlockLayer(BasicBlock, 512, 2, 2, 256)
#         self.layer2 = ResOutputLayer(BasicBlock, num_classes)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         return out

class ResNet34_Extractor(nn.Module):
    def __init__(self):
        super(ResNet34_Extractor, self).__init__()
        self.layer0 = ResInputLayer()
        self.layer1 = ResBlockLayer(BasicBlock, 64, 3, 1, 64)
    def forward(self, out):
        out = self.layer0(out)
        out = self.layer1(out)
        return out


class ResNet34_Classifer(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34_Classifer, self).__init__()
        self.layer0 = ResBlockLayer(BasicBlock, 128, 4, 2, 64)
        self.layer1 = ResBlockLayer(BasicBlock, 256, 6, 2, 128)
        self.layer2 = ResBlockLayer(BasicBlock, 512, 3, 2, 256)
        self.layer3 = ResOutputLayer(BasicBlock, num_classes)
    def forward(self, out):
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


# class ResNet34_Extractor(nn.Module):
#     def __init__(self):
#         super(ResNet34_Extractor, self).__init__()
#         self.layer0 = ResInputLayer()
#         self.layer1 = ResBlockLayer(BasicBlock, 64, 3, 1, 64)
#         self.layer2 = ResBlockLayer(BasicBlock, 128, 4, 2, 64)
#     def forward(self, out):
#         out = self.layer0(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         return out


# class ResNet34_Classifer(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet34_Classifer, self).__init__()
        
#         self.layer1 = ResBlockLayer(BasicBlock, 256, 6, 2, 128)
#         self.layer2 = ResBlockLayer(BasicBlock, 512, 3, 2, 256)
#         self.layer3 = ResOutputLayer(BasicBlock, num_classes)
#     def forward(self, out):
        
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         return out






class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(numclass=10):
    return ResNet(BasicBlock, [2,2,2,2], numclass)

def ResNet34(numclass=10):
    return ResNet(BasicBlock, [3,4,6,3], numclass)