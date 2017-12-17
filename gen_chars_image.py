#!/usr/bin/env python3

__all__ = (
    'generate_ims',
)

import math
import os
import random

import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common


def make_char_ims(output_height, font):
    font_size = output_height * 4
    font = ImageFont.truetype(font, font_size)
    height = max(font.getsize(d)[1] for d in common.CHARS)
    for c in common.CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.


def get_all_font_char_ims(out_height):
    # 一次生成所有字体的单个数字、英文字符
    result = []
    for font in common.fonts:
        result.append(dict(make_char_ims(out_height, font)))
    return result


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[c, 0., s],
                   [0., 1., 0.],
                   [-s, 0., c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[1., 0., 0.],
                   [0., c, -s],
                   [0., s, c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[c, -s, 0.],
                   [s, c, 0.],
                   [0., 0., 1.]]) * M

    return M


def get_affine_transform():
    # X 轴旋转
    pitch = random.uniform(-0.2, 0.2)
    # Y 轴旋转
    yaw = random.uniform(-0.5, 0.5)
    # Z 轴旋转
    roll = random.uniform(-0.05, 0.05)

    tran_x = random.uniform(-0.3, 0.5)
    tran_y = random.uniform(-0.3, 0.5)

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M = np.hstack([M, [[tran_x], [tran_y]]])
    return M


def generate_code():
    f = ""
    for i in range(common.LENGTH):
        f = f + random.choice(common.CHARS)
    return f


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.5) * font_height
    v_padding = random.uniform(0.2, 0.5) * font_height
    spacing = font_height * random.uniform(-0.01, 0.05)
    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color = random.uniform(0.5, 0.95)

    text_mask = np.zeros(out_shape)

    x = h_padding
    y = v_padding
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = text_color * text_mask

    return plate, code


def generate_rand_bg():
    bg = np.random.random_integers(
        0, 25, (common.OUTPUT_HEIGHT, common.OUTPUT_WIDTH)) / 255

    bg = cv2.GaussianBlur(bg, (7, 7), 0)

    return bg


def generate_im(char_ims):
    bg = generate_rand_bg()

    plate, code = generate_plate(common.FONT_HEIGHT, char_ims)
    plate = cv2.resize(plate, (common.OUTPUT_WIDTH, common.OUTPUT_HEIGHT))

    M = get_affine_transform()

    plate = cv2.warpAffine(
        plate, M, (common.OUTPUT_WIDTH, common.OUTPUT_HEIGHT))

    out = plate

    # add random noise
    out += np.random.normal(scale=0.02, size=out.shape)

    out = np.clip(out, 0., 1.)
    return out, code


def generate_ims(num_images):
    """
    Generate a number of number plate images.

    :param num_images:
        Number of images to generate.

    :return:
        Iterable of number plate images.

    """
    char_ims = get_all_font_char_ims(common.FONT_HEIGHT)
    for i in range(num_images):
        yield generate_im(random.choice(char_ims))


if __name__ == "__main__":
    dirs = ["test", "train"]
    size = {"test": common.TEST_SIZE, "train": common.TRAIN_SIZE}
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        im_gen = generate_ims(size.get(dir_name))
        for img_idx, (im, c) in enumerate(im_gen):
            fname = "{}/{:08d}_{}.png".format(dir_name, img_idx, c)
            print('\'' + fname + '\',')
            cv2.imwrite(fname, im * 255)
