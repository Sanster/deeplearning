#!/usr/bin/env python3

import glob
import numpy
import cv2
import numpy as np

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = list(DIGITS + LETTERS)
CHAR_SET_LENGTH = len(CHARS)

OUTPUT_CHAR_LENGTH = 3
OUTPUT_HEIGHT = 28
OUTPUT_WIDTH = 40

fonts = ["fonts/times.ttf", "fonts/Arial.ttf",
         "fonts/WKGOJanb.TTF", "fonts/miso-chunky.otf",
         "fonts/msyh.ttf"]

FONT_HEIGHT = 32  # Pixel size to which the chars are resized

BATCH_SIZE = 60
BATCHES = 1000
TRAIN_SIZE = BATCH_SIZE * BATCHES
TEST_SIZE = 100
