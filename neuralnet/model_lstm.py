from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

import cwt_ops
import subnets_ops


def model_lstm_v1(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):

    with tf.name_scope(name):

        wavelets, frequencies = cwt_ops.complex_morlet_wavelets(
            fc_array=params["fc_array"],
            fb_array=params["fb_array"],
            fs=params["fs"],
            lower_freq=params["lower_freq"],
            upper_freq=["upper_freq"],
            n_scales=params["n_scales"])

        cwt_sequence = cwt_ops.cwt_layer(
            inputs=input_sequence,
            wavelets=wavelets,
            border_crop=int(params["border_sec"]*params["fs"]),
            stride=params["time_stride"],
            frequencies=frequencies,
            flattening=True)

        # Normalize CWT
        inputs_cwt_bn = tf.layers.batch_normalization(inputs=cwt_sequence, training=training, reuse=reuse, name="bn_1")

        #h_t = None

        # Final Classification
        #with tf.variable_scope("output"):
        #    logits = tf.layers.dense(inputs=h_t, units=2, activation=None, name="logits", reuse=reuse)
        #    predictions = tf.nn.softmax(logits, axis=-1,  name="softmax")

        logits = None
        predictions = None

    return logits, predictions
