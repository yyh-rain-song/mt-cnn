import torch
import torch.utils.data
import torchvision
from loader import *
import os
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot



class Model(FCRN):
    def __init__(self):
        super(Model, self).__init__()
        self.conv4 = torch.nn.Conv2d(in_channels=1, kernel_size=3, out_channels=1, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.drop(x)

        x = self.conv3(x)
        x = self.relu(x)
        depth = self.upsample(x)

        level = self.conv4(depth)

        return depth, level
