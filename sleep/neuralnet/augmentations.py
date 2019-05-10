from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rescale_normal(feat, probability, std):
    with tf.variable_scope('rescale_normal'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat * tf.random.normal(
                [], mean=1.0, stddev=std),
            lambda: feat
        )
    return new_feat


def gaussian_noise(feat, probability, std):
    """Noise is relative to each value"""
    with tf.variable_scope('gaussian_noise'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat * (1.0 + tf.random.normal(
                tf.shape(feat), mean=0.0, stddev=std)),
            lambda: feat
        )
    return new_feat


def rescale_uniform(feat, probability, intensity):
    with tf.variable_scope('rescale_uniform'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat * tf.random.uniform(
                [], 1.0 - intensity, 1.0 + intensity),
            lambda: feat
        )
    return new_feat
