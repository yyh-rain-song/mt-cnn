from model import Model
import torch
from config import Config as cf
import torchvision
from weights import load_weights

class Trainer:
    def __init__(self, model, optimizer, epoch=50, training=True, use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.training = training
        self.epoch = epoch
        self.batch_pointer = 1
        self.use_cuda = use_cuda

    def get_batch_data(self):
        bx = torch.load(cf.data_path+"/"+"bx_{0}.th".format(self.batch_pointer))
        by = torch.load(cf.data_path+"/"+"by_{0}.th".format(self.batch_pointer))
        self.batch_pointer += 1
        self.batch_pointer = self.batch_pointer % 120 + 1
        if self.use_cuda:
            bx = bx.cuda()
            by = by.cuda()
        return bx, by

    def loss_func(self, y_depth, y_level, y, use_only_level=True):
        loss_depth = torch.nn.MSELoss()
        loss_level = torch.nn.CrossEntropyLoss()
        loss = loss_level(input=y_level, target=y[:, 2, :, :].long())
        if not use_only_level:
            loss += loss_depth(input=y_depth.float(),target=y[:, 1, :,:].float())
        return loss

    def train(self, use_only_level=False):
        # file = open("output.txt", "w")
        self.model.train()
        for i in range(0, self.epoch):
            x, y = self.get_batch_data()
            y_depth, y_level = self.model(x)
            shape = y_depth.shape
            y_depth = torch.reshape(torch.Tensor(y_depth), (-1, shape[2], shape[3]))
            y_level = torch.reshape(torch.Tensor(y_level), (-1, shape[2], shape[3]))
            loss = self.loss_func(y_depth, y_level, y, use_only_level)
            print("Epoch:{}, Loss:{:.4f}".format(i, loss.data))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i == self.epoch-1:
                torch.save(y_depth, "depth_only_depth")
                torch.save(y_level, "depth_only_level")
        torch.save(self.model.state_dict(), "./model/depth_only_model.pth")
        print("model saved!")

    def evaluate(self, input_x, input_y, use_only_level=False):
        self.model.eval()
        print("EValuate!")
        y_depth, y_level = self.model(input_x)
        loss = self.loss_func(y_depth, y_level, input_y, use_only_level)
        print("Evaluate Loss:{:.4f}".format(loss.data), file=open("eval_depth_only.txt", "w"))
        torch.save(y_depth, "eval_depth_only_depth")
        torch.save(y_level, "eval_depth_only_level")
        return y_depth, y_level

def main():
    # model = Model(cf.segment_class, cf.level_class, cf.image_scale)
    # batch_size = 16
    dtype = torch.cuda.FloatTensor
    weights_file = "NYU_ResNet-UpProj.npy"
    print("Loading model......")
    model = Model()
    #resnet = torchvision.models.resnet50(pretrained=True)
    resnet = torchvision.models.resnet50()
    resnet.load_state_dict(torch.load('/home/xuqingyao/Multi_CNN/model/resnet50.pth'))
    #resnet.load_state_dict(torch.load('/home/xpfly/nets/ResNet/resnet50-19c8e357.pth'))
    print("resnet50 loaded.")
    resnet50_pretrained_dict = resnet.state_dict()

    model.load_state_dict(load_weights(model, weights_file, dtype))
    """
    print('\nresnet50 keys:\n')
    for key, value in resnet50_pretrained_dict.items():
        print(key, value.size())
    """
    #model_dict = model.state_dict()
    """
    print('\nmodel keys:\n')
    for key, value in model_dict.items():
        print(key, value.size())

    print("resnet50.dict loaded.")
    """
    # load pretrained weights
    #resnet50_pretrained_dict = {k: v for k, v in resnet50_pretrained_dict.items() if k in model_dict}
    print("resnet50_pretrained_dict loaded.")
    """
    print('\nresnet50_pretrained keys:\n')
    for key, value in resnet50_pretrained_dict.items():
        print(key, value.size())
    """
    #model_dict.update(resnet50_pretrained_dict)
    print("model_dict updated.")
    """
    print('\nupdated model dict keys:\n')
    for key, value in model_dict.items():
        print(key, value.size())
    """
    #model.load_state_dict(model_dict)
    print("model_dict loaded.")
    # print("========== data has been load! ==========")
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("No cuda QAQ")
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=0.001), epoch=600, use_cuda=torch.cuda.is_available())
    trainer.train()

import matplotlib.pyplot as plt
if __name__ == '__main__':
    main()