#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

import common


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def code_list_to_num_list(code):
    return [common.CHARS.index(x) for x in code]


def code_to_one_hot(code):
    code_num = code_list_to_num_list(code)
    code_one_hot = np.zeros(
        (common.CHAR_SET_LENGTH * common.OUTPUT_CHAR_LENGTH))
    for i in range(common.OUTPUT_CHAR_LENGTH):
        num = code_num[i]
        code_one_hot[i * common.CHAR_SET_LENGTH + num] = 1

    # print("code num")
    # print(code_num)
    # print("code one hot")
    # print(code_one_hot)

    return code_one_hot


def one_hot_to_code(one_hot_code):
    reshaped = np.reshape(one_hot_code, (-1, common.OUTPUT_CHAR_LENGTH,
                                         common.CHAR_SET_LENGTH))
    # print("reshaped")
    # print(reshaped)
    # print("reshaped argmax")

    ret = ''
    for n in np.argmax(reshaped, 2).flatten():
        ret += common.CHARS[n]
    return ret
