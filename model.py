import torch
import ResNet
from ResNet_Dconv import resnet_deconv
from Config import Config as cf
import numpy as np
from PIL import Image


class Model(torch.nn.Module):
    def __init__(self, segment_classes, level_classes, img_scale):
        super(Model, self).__init__()
        resnet = ResNet.resnet18()
        self.segment_classes = segment_classes
        self.level_classes = level_classes
        # for param in resnet.parameters():
        #     param.requires_grad = False
        self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        plane = resnet.fc.in_features
        self.deconv1 = resnet_deconv(inplanes=plane, layers=[2, 2, 2, 2], out_classes=segment_classes, init_scale=img_scale)
        self.deconv2 = resnet_deconv(inplanes=plane, layers=[2, 2, 2, 2], out_classes=1, init_scale=img_scale)
        self.conv = torch.nn.Conv2d(in_channels=segment_classes+1, kernel_size=3, out_channels=level_classes, padding=1)
        self.droutput1 = torch.nn.Dropout(p=0.8)
        self.droutput2 = torch.nn.Dropout(p=0.8)
        # self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        # self.sigmoid3 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        seg = self.deconv1(self.droutput1(x))
        # seg = self.sigmoid1(seg)
        depth = self.deconv2(self.droutput2(x))
        depth = self.sigmoid2(depth)
        level = torch.cat((seg, depth), dim=1)
        level = self.conv(level)
        # level = self.sigmoid3(level)*3+1
        return seg, depth, level


class Trainer:
    def __init__(self, model, optimizer, epoch=50, training=True, use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.training = training
        self.epoch = epoch
        self.batch_pointer = 1
        self.use_cuda = use_cuda

    def get_batch_data(self):
        pt = int((self.batch_pointer+1)/2)
        bx = torch.load(cf.data_path+"/"+"bx_{0}.th".format(pt))
        by = torch.load(cf.data_path+"/"+"by_{0}.th".format(pt))
        self.batch_pointer += 1
        self.batch_pointer = self.batch_pointer % 1440 + 1
        if self.batch_pointer % 2 == 0:
            bx = bx[:8].clone()
            by = by[:8].clone()
        else:
            bx = bx[8:].clone()
            by = by[8:].clone()
        if self.use_cuda:
            bx = bx.cuda()
            by = by.cuda()
        return bx, by

    def loss_func(self, y_seg, y_depth, y_level, y, use_only_level=False):
        loss_depth = torch.nn.MSELoss()
        loss_level = torch.nn.CrossEntropyLoss()
        loss_seg = torch.nn.CrossEntropyLoss()
        loss = 2*loss_level(input=y_level, target=y[:, 2, :, :].long())
        if not use_only_level:
            loss += 5*loss_seg(input=y_seg, target=y[:, 0, :, :].long()) \
                    + loss_depth(input=y_depth.float(),target=y[:, 1, :,:].float())
        return loss

    def train(self, use_only_level=False, init_from_exist=False):
        if init_from_exist:
            dic = torch.load('./model/model.pth')
            self.model.load_state_dict(dic)
        self.model.train()
        for i in range(0, self.epoch):
            x, y = self.get_batch_data()
            y_seg, y_depth, y_level = self.model(x)
            loss = self.loss_func(y_seg, y_depth, y_level, y, use_only_level)
            if i % 10 == 0:
                print("Epoch:{}, Loss:{:.4f}".format(i, loss.data))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 1300 == 0:
                self.valid()
            if i == self.epoch-1:
                torch.save(y_level.clone().detach(), "d_level.th")
                torch.save(y[:,2,:,:].clone().detach(), 'd_truth.th')
        torch.save(self.model.state_dict(), "./model/model.pth")
        print("model saved!")

    def evaluate(self, test_idx, prefix, use_only_level=False, load_path=False):
        if load_path is True:
            dic = torch.load('./model/model.pth')
            self.model.load_state_dict(dic)
        bx = torch.load(cf.data_path + "/" + "bx_{0}.th".format(test_idx))
        by = torch.load(cf.data_path + "/" + "by_{0}.th".format(test_idx))
        if self.use_cuda:
            bx = bx.cuda()
            by = by.cuda()
        bx1 = bx[:8].clone()
        by1 = by[:8].clone()

        bx2 = bx[8:].clone()
        by2 = by[8:].clone()

        y_seg, y_depth, y_level = self.model(bx1)
        loss = self.loss_func(y_seg, y_depth, y_level, by1, use_only_level)
        torch.save(y_level, '/Disk2/yonglu/1/'+prefix+str(test_idx)+"_0.npy")
        y_seg, y_depth, y_level = self.model(bx2)
        loss += self.loss_func(y_seg, y_depth, y_level, by2, use_only_level)
        torch.save(y_level, '/Disk2/yonglu/1/' + prefix + str(test_idx) + "_1.npy")
        return loss.data/2

    def valid(self):
        self.model.eval()
        loss = 0
        idx = 0
        for i in range(651, 721):
            loss += self.evaluate(i, 'valid_')
            idx += 1
        loss = loss/idx
        self.model.train()
        print('========== Evaluate ! loss {0} ============'.format(loss))

    def test(self):
        self.model.eval()
        loss = 0
        idx = 0
        for i in range(721, 794):
            loss += self.evaluate(i, 'test_')
            idx += 1
        loss = loss/idx
        self.model.train()
        print('========== Test ! loss {0} ============'.format(loss))