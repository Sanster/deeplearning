#!/usr/bin/env python3

import tensorflow as tf

my_image = tf.zeros([10, 299, 299, 3])
r = tf.rank(my_image)

with tf.Session() as sess:
    print(sess.run(r))
