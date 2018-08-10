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


def lstm_base_model(input_sequence,
                    num_dirs,
                    params,
                    training,
                    reuse,
                    name):
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):
        cwt_sequence = subnets_ops.cwt_time_stride_layer(input_sequence, params, name="cwt")
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, use_in_bn=True, num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type+"_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, use_in_bn=True, num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type+"_2")
        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")
        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2, use_in_bn=True,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, 400, 2]
    return logits, predictions


def lstm_model(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model(input_sequence, 1, params, training, reuse, name)
    return logits, predictions


def blstm_model(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model(input_sequence, 2, params, training, reuse, name)
    return logits, predictions


def lstm_base_model_v2(input_sequence,
                       num_dirs,
                       params,
                       training,
                       reuse,
                       name):
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):
        # Regular BN at the end of CWT
        cwt_sequence = subnets_ops.cwt_time_stride_layer(input_sequence, params, name="cwt", use_out_bn=True)

        # No BN, Dropout on both lstm layers
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")
        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, num_dirs=num_dirs, use_in_drop=True,
                                              drop_rate=params["drop_rate"],
                                              training=training, reuse=reuse, name=lstm_type + "_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, num_dirs=num_dirs, use_in_drop=True,
                                              drop_rate=params["drop_rate"],
                                              training=training, reuse=reuse, name=lstm_type + "_2")
        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        # No dropout on the last FC layer
        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, 400, 2]
    return logits, predictions


def lstm_model_v2(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v2(input_sequence, 1, params, training, reuse, name)
    return logits, predictions


def blstm_model_v2(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v2(input_sequence, 2, params, training, reuse, name)
    return logits, predictions
