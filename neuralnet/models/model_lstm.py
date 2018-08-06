from __future__ import division
from __future__ import print_function
import tensorflow as tf

from models import subnets_ops


def dummy_model(
        input_sequence,  # [batch, 4400]
        params,
        training=False,
        reuse=False,
        name="model"):

    with tf.variable_scope(name):
        cwt_sequence = subnets_ops.cwt_time_stride_layer(input_sequence, params, name="cwt")
        out_seq = subnets_ops.conv_layer(cwt_sequence, filters=2, kernel_size=(32, 1), padding="valid",
                                         use_bn=True, training=training, reuse=reuse, name="conv_bn")
        # out [batch, 1, 400, 2]
        logits = tf.squeeze(out_seq, axis=[1], name="squeeze")  # out [batch, 400, 2]
        predictions = tf.nn.softmax(logits, axis=-1, name="softmax")
        tf.summary.histogram("predictions", predictions)
    return logits, predictions


def lstm_model(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):

    with tf.variable_scope(name):
        cwt_sequence = subnets_ops.cwt_time_stride_layer(input_sequence, params, name="cwt")

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
        cwt_sequence = subnets_ops.cwt_time_stride_layer(input_sequence, params, name="cwt")

        # Normalize CWT
        inputs_cwt_bn = tf.layers.batch_normalization(
            inputs=cwt_sequence, training=training, reuse=reuse, name="bn_1")
        # Output shape: [batch, 32, 400, 4]

        # TODO: implement BLSTM of two layers and 256 units each layer/direction
        logits = None
        predictions = None
    # Need shape [batch, 400, 2]
    return logits, predictions
