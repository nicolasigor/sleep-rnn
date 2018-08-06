from __future__ import division
from __future__ import print_function
import tensorflow as tf

from models import cwt_ops


def dummy_model(
        input_sequence,  # [batch, 4400]
        params,
        training=False,
        reuse=False,
        name="model"):

    with tf.variable_scope(name):

        wavelets, _ = cwt_ops.complex_morlet_wavelets(
            fb_array=params["fb_array"],
            fs=params["fs"],
            lower_freq=params["lower_freq"],
            upper_freq=params["upper_freq"],
            n_scales=params["n_scales"],
            flattening=True)

        cwt_sequence = cwt_ops.cwt_layer(
            inputs=input_sequence,
            wavelets=wavelets,
            border_crop=params["border_size"],
            stride=params["time_stride"])

        # Normalize CWT
        inputs_cwt_bn = tf.layers.batch_normalization(
            inputs=cwt_sequence, training=training, reuse=reuse, name="bn_1")

        out_seq = tf.layers.conv2d(inputs=inputs_cwt_bn, filters=2, kernel_size=(32, 1),
                                   padding="valid", reuse=reuse)  # out [batch, 1, 400, 2]
        logits = tf.squeeze(out_seq, axis=[1], name="squeeze")  # out [batch, 400, 2]
        predictions = tf.nn.softmax(logits, axis=-1, name="softmax")

    return logits, predictions


def lstm_model(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):

    with tf.variable_scope(name):

        wavelets, _ = cwt_ops.complex_morlet_wavelets(
            fb_array=params["fb_array"],
            fs=params["fs"],
            lower_freq=params["lower_freq"],
            upper_freq=params["upper_freq"],
            n_scales=params["n_scales"],
            flattening=True)

        cwt_sequence = cwt_ops.cwt_layer(
            inputs=input_sequence,
            wavelets=wavelets,
            border_crop=params["border_size"],
            stride=params["time_stride"])

        # Normalize CWT
        inputs_cwt_bn = tf.layers.batch_normalization(
            inputs=cwt_sequence, training=training, reuse=reuse, name="bn_1")
        # Output shape: [batch, 32, 400, 4]

        # TODO: implement LSTM of two layers and 256 units each layer
        logits = None
        predictions = None
    # Need shape [batch, 400, 2]
    return logits, predictions


def blstm_model(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):

    with tf.variable_scope(name):

        wavelets, _ = cwt_ops.complex_morlet_wavelets(
            fb_array=params["fb_array"],
            fs=params["fs"],
            lower_freq=params["lower_freq"],
            upper_freq=params["upper_freq"],
            n_scales=params["n_scales"],
            flattening=True)

        cwt_sequence = cwt_ops.cwt_layer(
            inputs=input_sequence,
            wavelets=wavelets,
            border_crop=params["border_size"],
            stride=params["time_stride"])

        # Normalize CWT
        inputs_cwt_bn = tf.layers.batch_normalization(
            inputs=cwt_sequence, training=training, reuse=reuse, name="bn_1")
        # Output shape: [batch, 32, 400, 4]

        # TODO: implement BLSTM of two layers and 256 units each layer/direction
        logits = None
        predictions = None
    # Need shape [batch, 400, 2]
    return logits, predictions
