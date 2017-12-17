#!/usr/bin/env python3

import glob
import numpy
import cv2
import numpy as np

DIGITS = "0123456789"
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
CHARS = list(DIGITS + LETTERS)
LENGTH = 6
OUTPUT_HEIGHT = 40
OUTPUT_WIDTH = 120

fonts = ["fonts/times.ttf", "fonts/Arial.ttf",
         "fonts/WKGOJanb.TTF", "fonts/miso-chunky.otf",
         "fonts/msyh.ttf"]

FONT_HEIGHT = 32  # Pixel size to which the chars are resized

BATCH_SIZE = 60
BATCHES = 100
TRAIN_SIZE = BATCH_SIZE * BATCHES
TEST_SIZE = 100
