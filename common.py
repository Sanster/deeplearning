#!/usr/bin/env python3

import glob
import numpy
import cv2
import numpy as np

DIGITS = "0123456789"
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
CHARS = list(DIGITS + LETTERS)
LENGTH = 6
# height * width
OUTPUT_SHAPE = (40, 120)

BATCH_SIZE = 60
BATCHES = 100
TRAIN_SIZE = BATCH_SIZE * BATCHES
TEST_SIZE = 100
