#!/usr/bin/env python3

import numpy as np
import glob
import cv2

import common

"""
    {"dirname":[(im,code)]}}
"""
data_set = {}


def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def code_list_to_num_list(code):
    return [common.CHARS.index(x) for x in code]


def code_to_one_hot(code):
    code_num = code_list_to_num_list(code)
    code_one_hot = np.zeros((common.CHAR_SET_LENGTH * common.OUTPUT_CHAR_LENGTH))
    for i in range(common.OUTPUT_CHAR_LENGTH):
        num = code_num[i]
        code_one_hot[i * common.CHAR_SET_LENGTH + num] = 1
    return code_one_hot


def load_data_set(data_dir):
    # load all imgs in data_dir into data_set
    result = list()
    fname_list = glob.glob(data_dir + "/*.png")
    for fname in sorted(fname_list):
        im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        im = np.reshape(im, (common.OUTPUT_HEIGHT, common.OUTPUT_WIDTH, 1))
        code = list(fname.split("/")[-1].split(".")[0].split("_")[1])
        index = int(fname.split("/")[-1].split("_")[0])

        # assum list index equals to image index
        result.append((im, code_to_one_hot(code)))
    data_set[data_dir] = result


def get_data_set(data_dir, start_index=None):
    if data_dir not in data_set.keys():
        load_data_set(data_dir)

    if start_index is None:
        return unzip(data_set[data_dir])
    else:
        return unzip(data_set[data_dir][start_index: start_index + common.BATCH_SIZE])


if __name__ == "__main__":
    images, labels = get_data_set("./test")
    print(images.shape)
    print(labels.shape)
    print(labels)
