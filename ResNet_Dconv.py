import torch
import torch.nn as nn


def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if stride == 1:
        pad = 0
    else:
        pad = 1
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, padding=1, output_padding=pad)


def deconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    if stride == 1:
        pad = 0
    else:
        pad = 1
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding=pad)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = deconv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = deconv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = deconv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDeconv(nn.Module):

    def __init__(self, block, inplanes, out_classes, layers, init_scale=None, zero_init_residual=False, norm_layer=None):
        super(ResNetDeconv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.out_classes = out_classes
        self.dilation = 1
        self.init_scale = init_scale
        if init_scale is None:
            self.init_scale = (500, 600)

        self.layer1 = self._make_layer(block, 512, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        scale = (int(init_scale[0]/2), int(init_scale[1]/2))
        self.unpooling = nn.Upsample(size=scale)
        self.deconv1 = nn.ConvTranspose2d(64, out_classes, kernel_size=7, stride=2, padding=3, output_padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.unpooling(x)
        x = self.deconv1(x)
        return x

    forward = _forward


def resnet_deconv(inplanes, layers, out_classes, block=None, init_scale=None):
    if block is None:
        block = Bottleneck
    model = ResNetDeconv(block, inplanes=inplanes, out_classes=out_classes, layers=layers, init_scale=init_scale)
    return model
