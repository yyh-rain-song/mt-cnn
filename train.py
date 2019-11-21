from model import Model, Trainer
import torch
from Config import Config as cf


def main():
    model = Model(cf.segment_class, cf.level_class, cf.image_scale)
    print("========== data has been load! ==========")
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("No cuda QAQ")
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=0.01), epoch=10, use_cuda=torch.cuda.is_available())
    trainer.train()
    # trainer.evaluate(X_test, Y_test)

    # seg, depth, level = trainer.evaluate(None)
    # torch.save(seg, "d_seg")
    # torch.save(depth, "d_depth")
    # torch.save(level, "d_level")


import matplotlib.pyplot as plt
if __name__ == '__main__':
    main()