from __future__ import division
from __future__ import print_function
import tensorflow as tf

from models import subnets_ops


def lstm_base_model(input_sequence,
                    num_dirs,
                    params,
                    training,
                    reuse,
                    name):
    # This model is CWT-LSTM-sigmoid with BN
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):
        # Regular BN at the end of CWT
        cwt_sequence = subnets_ops.cwt_local_stride_layer(input_sequence, params, name="cwt", use_out_bn=True)

        # Prepare for LSTM
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        # No dropout, regular BN at input (except first lstm because it already exists a BN before)
        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type+"_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, use_in_bn=True, num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type+"_2")

        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2, use_in_bn=True,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, time_len, 2]
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
    # This model is CWT-LSTM-sigmoid with BN only after CWT, and dropout at the input of both LSTMs
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):
        # Regular BN at the end of CWT
        cwt_sequence = subnets_ops.cwt_local_stride_layer(input_sequence, params, name="cwt", use_out_bn=True)

        # Prepare for LSTM
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        # No BN, Dropout on both lstm layers
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
    # Need shape [batch, time_len, 2]
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


'''
def lstm_base_model_v3(input_sequence,
                       num_dirs,
                       params,
                       training,
                       reuse,
                       name):
    # This model is CWT-CNN-LSTM-sigmoid with BN
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):
        # Less stride because of CNN stage
        cwt_sequence = subnets_ops.cwt_local_stride_layer(input_sequence, params, stride_reduction_factor=1/4,
                                                          name="cwt")
        # CNN stage, input shape [batch, time_len, n_scales, n_spectrograms]. BN at input
        cnn_sequence = subnets_ops.bn_conv3x3_block(cwt_sequence, 32, training=training,
                                                    reuse=reuse, name="conv_block_1")
        cnn_sequence = subnets_ops.bn_conv3x3_block(cnn_sequence, 64, training=training,
                                                    reuse=reuse, name="conv_block_2")

        # Prepare for LSTM
        temp_sequence = subnets_ops.sequence_flatten_layer(cnn_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        # No dropout, regular BN at input
        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, use_in_bn=True, num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type + "_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, use_in_bn=True, num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type + "_2")

        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2, use_in_bn=True,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, time_len, 2]
    return logits, predictions


def lstm_model_v3(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v3(input_sequence, 1, params, training, reuse, name)
    return logits, predictions


def blstm_model_v3(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v3(input_sequence, 2, params, training, reuse, name)
    return logits, predictions
'''
