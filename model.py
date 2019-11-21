import torch
import ResNet
from ResNet_Dconv import resnet_deconv
from Config import Config as cf
import numpy as np
from PIL import Image


class Model(torch.nn.Module):
    def __init__(self, segment_classes, level_classes, img_scale):
        super(Model, self).__init__()
        resnet = ResNet.resnet50()
        self.segment_classes = segment_classes
        self.level_classes = level_classes
        for param in resnet.parameters():
            param.requires_grad = False
        self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        plane = resnet.fc.in_features
        self.deconv1 = resnet_deconv(inplanes=plane, layers=[2, 2, 2, 2], out_classes=segment_classes, init_scale=img_scale)
        self.deconv2 = resnet_deconv(inplanes=plane, layers=[2, 2, 2, 2], out_classes=1, init_scale=img_scale)
        self.conv = torch.nn.Conv2d(in_channels=segment_classes+1, kernel_size=3, out_channels=level_classes, padding=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.sigmoid3 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        seg = self.deconv1(x)
        seg = self.sigmoid1(seg)
        depth = self.deconv2(x)
        depth = self.sigmoid2(depth)
        level = torch.cat((seg, depth), dim=1)
        level = self.conv(level)
        level = self.sigmoid3(level)
        return seg, depth, level


class Trainer:
    def __init__(self, model, optimizer, epoch=50, training=True, use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.training = training
        self.epoch = epoch
        self.batch_pointer = 0
        self.use_cuda = use_cuda

    def get_batch_data(self):
        bx = torch.load(cf.data_path+"/"+"bx_{0}.th".format(self.batch_pointer))
        by = torch.load(cf.data_path+"/"+"by_{0}.th".format(self.batch_pointer))
        self.batch_pointer += 1
        self.batch_pointer = self.batch_pointer % 60
        if self.use_cuda:
            bx = bx.cuda()
            by = by.cuda()
        return bx, by

    def loss_func(self, y_seg, y_depth, y_level, y, use_only_level=False):
        loss_depth = torch.nn.MSELoss()
        loss_level = torch.nn.CrossEntropyLoss()
        loss_seg = torch.nn.CrossEntropyLoss()
        loss = loss_level(input=y_level, target=y[:, 2, :, :].long())
        if not use_only_level:
            loss += loss_seg(input=y_seg, target=y[:, 0, :, :].long()) \
                    + loss_depth(input=y_depth.float(),target=y[:, 1, :,:].float())
        return loss

    def train(self, use_only_level=False):
        file = open("output.txt", "w")
        self.model.train()
        for i in range(0, self.epoch):
            x, y = self.get_batch_data()
            y_seg, y_depth, y_level = self.model(x)
            loss = self.loss_func(y_seg, y_depth, y_level, y, use_only_level)
            print("Epoch:{}, Loss:{:.4f}".format(i, loss.data), file=file)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i == self.epoch-1:
                torch.save(y_seg, "d_seg")
                torch.save(y_depth, "d_depth")
                torch.save(y_level, "d_level")
        torch.save(self.model.state_dict(), "./model/model.pth")
        print("model saved!", file=file)

    def evaluate(self, input_x, input_y, use_only_level=False):
        self.model.eval()
        y_seg, y_depth, y_level = self.model(input_x)
        loss = self.loss_func(y_seg, y_depth, y_level, input_y, use_only_level)
        print("Evaluate Loss:{:.4f}".format(loss.data), file=open("eval.txt", "w"))
        torch.save(y_seg, "eval_seg")
        torch.save(y_depth, "eval_depth")
        torch.save(y_level, "eval_level")
        return y_seg, y_depth, y_level
