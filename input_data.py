#!/usr/bin/env python3

import numpy as np
import glob
import cv2

import common

"""
    {"dirname":[(im,code)]}}
"""
data_set = {}


def load_data_set(data_dir):
    # load all imgs in data_dir into data_set
    result = list()
    fname_list = glob.glob(data_dir + "/*.png")
    for fname in sorted(fname_list):
        im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        code = list(fname.split("/")[-1].split("_")[1])
        index = int(fname.split("/")[-1].split("_")[0])
        result.append((im, code))
    data_set[data_dir] = result


def get_data_set(data_dir, start_index=None):
    if data_dir not in data_set.keys():
        load_data_set(data_dir)

    if start_index is None:
        return data_set[data_dir]
    else:
        return data_set[data_dir][start_index: common.BATCH_SIZE]
