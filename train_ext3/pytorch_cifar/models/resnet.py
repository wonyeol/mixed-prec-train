'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
#===== 
# import torch.nn as nn
# import torch.nn.functional as F 
import ext3.nn as nn
import torch.nn.init as init
from   ext3.util import _make_divisible
#=====

#=====
# EDITS.
# - F.{relu, avg_pool2d}.
# - t1 + t2.
# - t.view() ---> no edit.
#=====

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        #===== 
        self.relu1 = nn.ReLU()
        self.add = nn.Add()
        self.relu3 = nn.ReLU()
        #=====

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = self.add(out, self.shortcut(x))
        # out = F.relu(out)
        out = self.relu3(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        #===== 
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.add = nn.Add()
        self.relu4 = nn.ReLU()
        #=====

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out = self.add(out, self.shortcut(x))
        # out = F.relu(out)
        out = self.relu4(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width_mult=1.0, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        #===== 
        self.planes = [64, 128, 256, 512]

        # handle width_mult. (src: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
        self.width_mult    = width_mult
        self.round_nearest = 4 if width_mult <= 0.25 else 8
        self.in_planes, = [_make_divisible(v * self.width_mult, self.round_nearest) for v in (self.in_planes,)]
        self.planes     = [_make_divisible(v * self.width_mult, self.round_nearest) for v in  self.planes     ]
        #=====

        #===== 
        self.input = nn.Input()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.conv1 = nn.Conv2d(3,   self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.planes[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(self.planes[3] * block.expansion, num_classes)

        self.relu1 = nn.ReLU()
        self.avgpool2d = nn.AvgPool2d(4)
        #=====

        #===== 
        # src: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)        
        #=====

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #===== 
        x = self.input(x)
        #=====
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool2d(out)
        # out = out.view(out.size(0), -1)
        out = nn.view(out, (out.size(0), -1))
        out = self.linear(out)
        return out


# def ResNet18():
def ResNet18(width_mult=1.0, num_classes=10):
    # return ResNet(BasicBlock, [2, 2, 2, 2])
    return ResNet(BasicBlock, [2, 2, 2, 2], width_mult, num_classes)

# def ResNet34():
def ResNet34(width_mult=1.0, num_classes=10):
    # return ResNet(BasicBlock, [3, 4, 6, 3])
    return ResNet(BasicBlock, [3, 4, 6, 3], width_mult, num_classes)

# def ResNet50():
def ResNet50(width_mult=1.0, num_classes=10):
    # return ResNet(Bottleneck, [3, 4, 6, 3])
    return ResNet(Bottleneck, [3, 4, 6, 3], width_mult, num_classes)

# def ResNet101():
def ResNet101(width_mult=1.0, num_classes=10):
    # return ResNet(Bottleneck, [3, 4, 23, 3])
    return ResNet(Bottleneck, [3, 4, 23, 3], width_mult, num_classes)

# def ResNet152():
def ResNet152(width_mult=1.0, num_classes=10):
    # return ResNet(Bottleneck, [3, 8, 36, 3])
    return ResNet(Bottleneck, [3, 8, 36, 3], width_mult, num_classes)

# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())
#
# # test()
