#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import argparse
import cv2
import sys

import common
from input_data import *
from util import *

FLAGS = None


def cnn_net(x):
    """
    Args:
      x: an input tensor with the dimensions (N_examples, [h, w])
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 40x120 image
    # is down to 10x30x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 10 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 10 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Map the 1024 features to the length of char, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable(
            [1024, common.CHAR_SET_LENGTH * common.OUTPUT_CHAR_LENGTH])
        b_fc2 = bias_variable(
            [common.CHAR_SET_LENGTH * common.OUTPUT_CHAR_LENGTH])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def to_max(y):
    return tf.argmax(tf.reshape(y, [-1, common.CHAR_SET_LENGTH, common.OUTPUT_CHAR_LENGTH]), 2)


def main():
    x = tf.placeholder(tf.float32, shape=(
        None, common.OUTPUT_HEIGHT, common.OUTPUT_WIDTH, 1))

    y_ = tf.placeholder(
        tf.float32, [None, common.CHAR_SET_LENGTH * common.OUTPUT_CHAR_LENGTH])

    y_conv = cnn_net(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(
            1e-3).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(to_max(y_conv), to_max(y_))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    test_images, test_labels = get_data_set('./test')
    print("Finish loading test data")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(common.BATCHES):
            batch_images, batch_labels = get_data_set(
                './train', common.BATCH_SIZE)

            if i % 50 == 0 and i != 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: batch_images, y_: batch_labels})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            # train_step.run(feed_dict={x: batch_images, y_: batch_labels})
            _, train_loss = sess.run([train_step, cross_entropy], feed_dict={
                                     x: batch_images, y_: batch_labels})
            print("Batch: {}, Loss: {}".format(i, train_loss))

        print('test accuracy %g' % accuracy.eval(
            feed_dict={x: test_images, y_: test_labels}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./train/')
    parser.add_argument('--test_dir', type=str, default='./test/')
    FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    # 直接跑 main 和 tf.app.run 有什么区别？
    main()
