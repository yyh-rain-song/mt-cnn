import cv2
import numpy as np
import os


def get_ave_depth(segment, depth_mat, height, width):
    area = {}
    area_depth = {}
    for i in range(0, height):
        for j in range(0, width):
            dot = segment[i][j]
            color = dot[0]*256*256 + dot[1]*256 + dot[2]
            d = depth_mat[i][j]
            if color in area:
                area[color] += 1
                area_depth[color] += d
            else:
                area[color] = 1
                area_depth[color] = d

    area_ave = {}
    for color in area.keys():
        area_ave[color] = area_depth[color] / area[color]
    return area_ave


def paint_graph(area_ave, depth_level, height, width):
    new_image = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            dot = seg[i][j]
            color = dot[0]*256*256 + dot[1]*256 + dot[2]
            lv = 0
            while depth_level[lv] + 1e-5 <= area_ave[color]:
                lv += 1
            new_image[i][j] = color_level[lv]
    return new_image


seg_root = "./data/livingroom_segmentation"
dp_root = "./data/livingroom_depth"
target_root = "./data/livingroom_level"
files = os.listdir(seg_root)
for file in files:
    name = file.split(".")[0]
    print(name)
    seg = cv2.imread(seg_root + "/" + file, cv2.IMREAD_UNCHANGED)
    depth = np.load(dp_root + "/" + name + ".npy")
    color_level = [0, 85, 170, 255]
    h = seg.shape[0]
    w = seg.shape[1]

    average_depth = get_ave_depth(seg, depth, h, w)
    a_d = average_depth.values()
    a_d = np.array(list(a_d))
    d_l = [(a_d.max()-a_d.min())/3*i+a_d.min() for i in range(0, 4)]
    level_img = paint_graph(average_depth, d_l, h, w)

    cv2.imwrite(target_root + "/" + file, level_img)

