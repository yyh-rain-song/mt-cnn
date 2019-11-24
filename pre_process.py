import json
# import xmltodict
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# def xmltojson(name, file):
#     dirc = "../SUNDATABASE/Annotations/l/living_room"
#     outputdirc = "../SUNDATABASE/Annotations/l/json"
#     xmlstr = file.read()
#     xmlparse = xmltodict.parse(xmlstr)
#     jsonstr = json.dumps(xmlparse,indent=1)
#     print(jsonstr, file=open(outputdirc+"/"+name+".json", "w"))


# def tojson(dirc):
#     files = os.listdir(dirc)
#     for file in files:
#         name = file.split('.')[0]
#         print(name)
#         f = open(dirc+"/"+file, "r")
#         xmltojson(name, f)


def read_json():
    with open("./data/class.json", 'r') as load_f:
        load_dict = json.load(load_f)
    keys = [tmp[0] for tmp in load_dict]
    for k in keys:
        print(k)


def deal_data(only_level=False):
    img_root = "../data/bedroom"
    level_root = "../data/levels"
    seg_root = "../data/seg_mat"
    deep_root = "../data/depth_new"

    wanted_size = (300, 200)
    files = os.listdir(img_root)
    images = None
    segmentation = None
    depth__ = None
    level__ = None

    for i in range(0, 2118):
        f_name = files[i].split('.')[0]
        img = cv2.imread(img_root + "/" + f_name + ".jpg")
        img = cv2.resize(img, dsize=wanted_size, interpolation=cv2.INTER_NEAREST)
        seg = np.load(seg_root+"/"+f_name+".npy")
        depth = np.load(deep_root+"/"+f_name+".npy")
        level = np.load(level_root+"/"+f_name+".npy")
        if i%16==0:
            if i!=0:
                print(i)
                X = images
                Y = torch.cat((segmentation, depth__, level__), dim=3)
                Y = Y.transpose(2, 3)
                Y = Y.transpose(1, 2)
                X = X.transpose(2, 3)
                X = X.transpose(1, 2)
                print(Y.shape)
                torch.save(X, "../data/data_batch/bx_{0}.th".format(int(i/16)))
                torch.save(Y, "../data/data_batch/by_{0}.th".format(int(i/16)))

            images = torch.reshape(torch.Tensor(img), (1, wanted_size[1], wanted_size[0], 3))
            segmentation = torch.reshape(torch.Tensor(seg), (1, wanted_size[1], wanted_size[0], 1))
            depth__ = torch.reshape(torch.Tensor(depth), (1, wanted_size[1], wanted_size[0], 1))
            level__ = torch.reshape(torch.Tensor(level), (1, wanted_size[1], wanted_size[0], 1))
        else:
            images = torch.cat((images, torch.reshape(torch.Tensor(img), (1, wanted_size[1], wanted_size[0], 3))))
            segmentation = torch.cat((segmentation, torch.reshape(torch.Tensor(seg), (1, wanted_size[1], wanted_size[0], 1))))
            depth__ = torch.cat((depth__, torch.reshape(torch.Tensor(depth), (1, wanted_size[1], wanted_size[0], 1))))
            level__ = torch.cat((level__, torch.reshape(torch.Tensor(level), (1, wanted_size[1], wanted_size[0], 1))))


def get_better_size():
    dirt = "../data/bedroom"
    images = os.listdir(dirt)
    l = []
    w = []
    for f in images:
        img = cv2.imread(dirt+"/"+f)
        l.append(img.shape[0])
        w.append(img.shape[1])
    l.sort()
    w.sort()
    for i in range(0, 21):
        print(l[i*100], end=' ')
    print("")
    for i in range(0, 21):
        print(w[i * 100], end=' ')


def make_test_batch():
    tx = torch.load('../data/data_batch/tx.th')
    ty = torch.load('../data/data_batch/ty.th')
    print(tx.shape)
    print(ty.shape)
    for i in range(0, int(tx.shape[0]/16)):
        frt = i*16
        bk = (i+1)*16
        tx_i = tx[frt:bk].clone().detach()
        ty_i = ty[frt:bk].clone().detach()
        torch.save(tx_i, "../data/data_batch/tx_{0}".format(i))
        torch.save(ty_i, "../data/data_batch/ty_{0}".format(i))


if __name__ == '__main__':
    make_test_batch()
    pass
