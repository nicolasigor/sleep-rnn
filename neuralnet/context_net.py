from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def context_net(inputs, use_batchnorm=True, training=False, reuse=False, name="context_net"):
    with tf.variable_scope(name):
        # First convolutional block
        c_1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, activation=tf.nn.relu,
                               padding="same", name="c_1", reuse=reuse)
        p_1 = tf.layers.max_pooling2d(inputs=c_1, pool_size=2, strides=2)

        # Second convolutional block
        if use_batchnorm:
            p_1 = tf.layers.batch_normalization(inputs=p_1, training=training, name="bn_1", reuse=reuse)
        c_2 = tf.layers.conv2d(inputs=p_1, filters=32, kernel_size=3, activation=tf.nn.relu,
                               padding="same", name="c_2", reuse=reuse)
        p_2 = tf.layers.max_pooling2d(inputs=c_2, pool_size=2, strides=2)

        # Third convolutional block
        if use_batchnorm:
            p_2 = tf.layers.batch_normalization(inputs=p_2, training=training, name="bn_2", reuse=reuse)
        c_3a = tf.layers.conv2d(inputs=p_2, filters=64, kernel_size=3, activation=tf.nn.relu,
                                padding="same", name="c_3a", reuse=reuse)
        if use_batchnorm:
            c_3a = tf.layers.batch_normalization(inputs=c_3a, training=training, name="bn_3", reuse=reuse)
        c_3b = tf.layers.conv2d(inputs=c_3a, filters=64, kernel_size=3, activation=tf.nn.relu,
                                padding="same", name="c_3b", reuse=reuse)
        p_3 = tf.layers.max_pooling2d(inputs=c_3b, pool_size=2, strides=2)

        # Fourth convolutional block
        if use_batchnorm:
            p_3 = tf.layers.batch_normalization(inputs=p_3, training=training, name="bn_4", reuse=reuse)
        c_4a = tf.layers.conv2d(inputs=p_3, filters=64, kernel_size=3, activation=tf.nn.relu,
                                padding="same", name="c_4a", reuse=reuse)
        if use_batchnorm:
            c_4a = tf.layers.batch_normalization(inputs=c_4a, training=training, name="bn_5", reuse=reuse)
        c_4b = tf.layers.conv2d(inputs=c_4a, filters=64, kernel_size=3, activation=tf.nn.relu,
                                padding="same", name="c_4b", reuse=reuse)
        p_4 = tf.layers.max_pooling2d(inputs=c_4b, pool_size=2, strides=2)

        # Flattening
        dim = np.prod(p_4.get_shape().as_list()[1:])
        c_t_flat = tf.reshape(p_4, shape=(-1, dim))
    return c_t_flat
