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
                                         use_in_bn=True, training=training, reuse=reuse, name="conv_bn")
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
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")
        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, use_in_bn=True,
                                              training=training, reuse=reuse, name="lstm_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, use_in_bn=True,
                                              training=training, reuse=reuse, name="lstm_2")
        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")
        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2, use_in_bn=True,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
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
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")
        blstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_dirs=2, num_units=256, use_in_bn=True,
                                               training=training, reuse=reuse, name="blstm_1")
        blstm_2 = subnets_ops.cudnn_lstm_layer(blstm_1, num_dirs=2, num_units=256, use_in_bn=True,
                                               training=training, reuse=reuse, name="blstm_2")

        temp_sequence = subnets_ops.undo_time_major_layer(blstm_2, name="undo_time_major")
        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2, use_in_bn=True,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, 400, 2]
    return logits, predictions
