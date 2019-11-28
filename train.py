from model import Model, Trainer
import torch
from Config import Config as cf


def main():
    model = Model(cf.segment_class, cf.level_class, cf.image_scale)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("No cuda QAQ")
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), cf.lr), epoch=cf.epoch, use_cuda=torch.cuda.is_available(),
                      loss_weight=cf.loss_weight, loss_func=3)
    trainer.train(init_from_exist=cf.import_model)
    trainer.test()


if __name__ == '__main__':
    main()