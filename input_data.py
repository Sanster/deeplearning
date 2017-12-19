#!/usr/bin/env python3

import numpy as np
import glob
import cv2

import common
import random

from util import *

"""
    {"dirname":[(im,code)]}}
"""
data_set = {}


def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def load_one_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    img = np.reshape(img, (common.OUTPUT_HEIGHT, common.OUTPUT_WIDTH, 1))
    code = list(img_path.split("/")[-1].split(".")[0].split("_")[1])
    index = int(img_path.split("/")[-1].split("_")[0])
    return img, code_to_one_hot(code)


def load_data_set(data_dir):
    # load all imgs in data_dir into data_set
    result = list()
    fname_list = glob.glob(data_dir + "/*.png")
    for fname in sorted(fname_list):
        img, code_one_hot = load_one_image(fname)
        # assum list index equals to image index
        result.append((img, code_one_hot))
    data_set[data_dir] = result


def get_data_set(data_dir, size=None):
    if data_dir not in data_set.keys():
        load_data_set(data_dir)

    if 'train' in data_dir:
        img_count = common.TRAIN_SIZE
    elif 'test' in data_dir:
        img_count = common.TEST_SIZE

    if size is None:
        return unzip(data_set[data_dir])
    else:
        random_data = []
        for i in range(size):
            ix = random.randint(0, img_count - 1)
            random_data.append(data_set[data_dir][ix])

        return unzip(random_data)


if __name__ == "__main__":
    images, labels = get_data_set("./test")
    print(images.shape)
    print(labels.shape)
    print(labels)
