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


def smalldata_rsz():
    for i in range(1, 7):
        img = cv2.imread('./image/{0}.jpg'.format(i), cv2.IMREAD_UNCHANGED)

        print('Original Dimensions : ', img.shape)

        scale_percent = 60  # percent of original size
        width = 128
        height = 160
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        print('Resized Dimensions : ', resized.shape)

        cv2.imshow("Resized image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("./image/{0}_norm.jpg".format(i), resized)


def get_name_from_dir(path):
    pictures = os.listdir(path)
    pictures = [tmp.split('.')[0] for tmp in pictures]
    return pictures


def kick_bad_data():
    img_root = "./data/livingroom"
    seg_root = "./data/livingroom_segmentation"
    depth_root = "./data/livingroom_depth"
    level_root = "./data/livingroom_level"

    img = get_name_from_dir(img_root)
    seg = get_name_from_dir(seg_root)
    depth = get_name_from_dir(depth_root)
    level = get_name_from_dir(level_root)

    i1 = 0
    i2 = 0
    i3 = 0
    for i in range(0, len(img)):
        txt = img[i]
        if seg[i1] == txt:
            i1 += 1
        else:
            st = str(txt) + " doesn't have segment"
            print(st)
        if depth[i2] == txt:
            i2 += 1
        else:
            print(str(txt) + " doesn't have depth")
        if level[i3] == txt:
            i3 += 1
        else:
            print(str(txt) + " doesn't have level")


def read_json():
    with open("./data/class.json", 'r') as load_f:
        load_dict = json.load(load_f)
    keys = [tmp[0] for tmp in load_dict]
    for k in keys:
        print(k)


def deal_data(only_level=False):
    global Y
    img_root = "./data/bedroom"
    level_root = "./data/levels"
    seg_root = "./data/seg_matrix"
    deep_root = "./data/depth"

    wanted_size = (300, 200)
    files = os.listdir(level_root)
    images = None
    segmentation = None
    depth__ = None
    level__ = None

    for i in range(0, 2115-800):
        f_name = files[i].split('.')[0]
        img = cv2.imread(img_root + "/" + f_name + ".jpg")
        img = cv2.resize(img, dsize=wanted_size, interpolation=cv2.INTER_NEAREST)
        seg = np.load(seg_root+"/"+f_name+".npy")
        depth = np.load(deep_root+"/"+f_name+".npy")
        level = np.load(level_root+"/"+f_name+".npy")
        if i%100==0:
            if i!=0:
                print(i)
                X = images
                Y = torch.cat((segmentation, depth__, level__), dim=3)
                Y = Y.transpose(2, 3)
                Y = Y.transpose(1, 2)
                X = X.transpose(2, 3)
                X = X.transpose(1, 2)
                print(Y.shape)
                torch.save(X, "./data/datax_{0}.th".format(i+800))
                torch.save(Y, "./data/datay_{0}.th".format(i+800))

            images = torch.reshape(torch.Tensor(img), (1, wanted_size[1], wanted_size[0], 3))
            segmentation = torch.reshape(torch.Tensor(seg), (1, wanted_size[1], wanted_size[0], 1))
            depth__ = torch.reshape(torch.Tensor(depth), (1, wanted_size[1], wanted_size[0], 1))
            level__ = torch.reshape(torch.Tensor(level), (1, wanted_size[1], wanted_size[0], 1))
        else:
            images = torch.cat((images, torch.reshape(torch.Tensor(img), (1, wanted_size[1], wanted_size[0], 3))))
            segmentation = torch.cat((segmentation, torch.reshape(torch.Tensor(seg), (1, wanted_size[1], wanted_size[0], 1))))
            depth__ = torch.cat((depth__, torch.reshape(torch.Tensor(depth), (1, wanted_size[1], wanted_size[0], 1))))
            level__ = torch.cat((level__, torch.reshape(torch.Tensor(level), (1, wanted_size[1], wanted_size[0], 1))))


def deal_pic(image, dic):
    w = image.shape[0]
    h = image.shape[1]
    matrix = np.zeros((w, h))
    for i in range(0, w):
        for j in range(0, h):
            color = image[i][j]
            # key = color[0]*256*256 + color[1]*256 + color[2]
            key = color
            if key in dic.keys():
                cls = dic[key]
                matrix[i, j] = cls
            else:
                print(color)
                cls = len(dic.keys())
                dic[key] = cls
                matrix[i, j] = cls
    return matrix


def change_seg():
    outdic = "./data/bedroom_level"
    indic = "./data/bedroom_level"
    files = os.listdir(indic)
    dictory = {}
    i = 0
    for img in files:
        i += 1
        img = img.split('.')[0]
        image = cv2.imread(indic + "/" + img + ".png")
        mtx = deal_pic(image, dictory)
        np.save(outdic+"/"+img+".tpy", mtx)
        if i == 5:
            return


def draw_img():
    depth = np.load('draw_it.npy')
    # print(depth)
    fig = plt.figure()
    ii = plt.imshow(depth, cmap='gray')
    plt.axis('off')
    fig.colorbar(ii)
    plt.savefig('level.png')


def batch_gen_data():
    # x = torch.zeros((2100, 3, 200, 300))
    # y = torch.zeros((2100, 3, 200, 300))
    # for i in range(1, 22):
    #     frt = (i-1)*100
    #     bk = i*100
    #     x[frt:bk] = torch.load("./data/data_concat/datax_{}.th".format(bk))
    #     y[frt:bk] = torch.load("./data/data_concat/datay_{}.th".format(bk))
    # torch.save(x, "data_x.th")
    # torch.save(y, "data_y.th")

    x = torch.load("data_x.th")
    y = torch.load("data_y.th")

    for i in range(0, 120):
        frt = i*16
        bk = (i+1)*16
        bx = x[frt:bk].clone()
        by = y[frt:bk].clone()
        torch.save(bx, "./data/data_batch/bx_{0}.th".format(i))
        torch.save(by, "./data/data_batch/by_{0}.th".format(i))
    tx = x[60*32:].clone()
    ty = y[60*32:].clone()
    torch.save(tx, "./data/data_batch/tx.th")
    torch.save(ty, "./data/data_batch/ty.th")


if __name__ == '__main__':
    batch_gen_data()
    pass
