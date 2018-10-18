from __future__ import division
from __future__ import print_function
import tensorflow as tf

from models import subnets_ops

'''
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
        cwt_sequence = subnets_ops.cwt_local_stride_layer(input_sequence, params, name="cwt", use_out_bn=True,
                                                          training=training)

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
        cwt_sequence = subnets_ops.cwt_local_stride_layer(input_sequence, params, name="cwt", log_transform=True,
                                                          use_out_bn=True, training=training)
        # Prepare for LSTM
        temp_sequence = subnets_ops.sequence_flatten_layer(cwt_sequence, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        # No BN, Dropout on both lstm layers
        # lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, num_dirs=num_dirs, use_in_drop=True,
        #                                       drop_rate=params["drop_rate"],
        #                                       training=training, reuse=reuse, name=lstm_type + "_1")
        # lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, num_dirs=num_dirs, use_in_drop=True,
        #                                       drop_rate=params["drop_rate"],
        #                                       training=training, reuse=reuse, name=lstm_type + "_2")
        #
        # temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        n_layers = 3
        num_units = 256
        lstm_out = temp_sequence
        for i in range(n_layers):
            # lstm_out = subnets_ops.cudnn_lstm_layer(lstm_out, num_units=num_units, num_dirs=num_dirs, use_in_drop=True,
            #                                         drop_rate=params["drop_rate"], training=training, reuse=reuse,
            #                                         name=lstm_type+"_"+str(i))
            if i==0:
                use_in_bn=False
            else:
                use_in_bn=True
            lstm_out = subnets_ops.cudnn_lstm_layer(lstm_out, num_units=num_units, num_dirs=num_dirs, use_in_bn=use_in_bn,
                                                    training=training, reuse=reuse, name=lstm_type+"_"+str(i+1))
        temp_sequence = subnets_ops.undo_time_major_layer(lstm_out, name="undo_time_major")

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


def lstm_base_model_v0(
        input_sequence,
        num_dirs,
        params,
        training,
        reuse,
        name):
    # Very simple model
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):

        cwt_sequence = subnets_ops.cwt_local_stride_layer(
            input_sequence, params, name="cwt", use_out_bn=False, training=training,
            use_avg_pool=False, log_transform=True, stride_reduction_factor=1/4)

        # Convolutional stage input shape [batch, time_len, n_scales, n_spectrograms]
        # out_cnn = cwt_sequence
        # out_cnn = tf.layers.conv2d(inputs=cwt_sequence, filters=16, kernel_size=3, activation=tf.nn.relu, stride=2,
        #                            padding="same", name="conv_1", reuse=reuse)
        # out_cnn = tf.layers.conv2d(inputs=out_cnn, filters=32, kernel_size=3, activation=tf.nn.relu, stride=2,
        #                            padding="same", name="conv_2", reuse=reuse)

        # out_cnn = subnets_ops.conv_layer(inputs=cwt_sequence, filters=16, strides=1, activation=tf.nn.relu,
        #                                  use_in_bn=True,
        #                                  training=training, reuse=reuse, name="conv_1a")
        # out_cnn = subnets_ops.conv_layer(inputs=out_cnn, filters=16, strides=2, activation=tf.nn.relu,
        #                                  use_in_bn=True,
        #                                  training=training, reuse=reuse, name="conv_1b")
        # out_cnn = subnets_ops.conv_layer(inputs=out_cnn, filters=32, strides=1, activation=tf.nn.relu,
        #                                  use_in_bn=True,
        #                                  training=training, reuse=reuse, name="conv_2a")
        # out_cnn = subnets_ops.conv_layer(inputs=out_cnn, filters=32, strides=2, activation=tf.nn.relu,
        #                                  use_in_bn=True,
        #                                  training=training, reuse=reuse, name="conv_2b")

        out_cnn = subnets_ops.conv_layer(inputs=cwt_sequence, filters=16, strides=2, activation=tf.nn.relu,
                                         use_in_bn=True,
                                         training=training, reuse=reuse, name="conv_1")
        out_cnn = subnets_ops.conv_layer(inputs=out_cnn, filters=16, strides=2, activation=tf.nn.relu,
                                         use_in_bn=True,
                                         training=training, reuse=reuse, name="conv_2")

        # Flattening
        temp_sequence = subnets_ops.sequence_flatten_layer(out_cnn, name="flatten")

        # Local contexttf.concat([[1], tf.shape(inputs)[1:]], axis=0)
        with tf.name_scope("short_context"):
            in_sh = tf.shape(temp_sequence)
            zero_pad_time = tf.zeros([in_sh[0], 1, in_sh[2]], tf.float32)
            past_values = tf.concat([zero_pad_time, temp_sequence[:, :-1, :]], 1)
            future_values = tf.concat([temp_sequence[:, 1:, :], zero_pad_time], 1)
            temp_sequence = tf.concat([past_values, temp_sequence, future_values], 2)

        # Prepare for LSTM
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=512, num_dirs=num_dirs,
                                              # use_in_bn=True,
                                              # use_in_drop=True, drop_rate=0.3,
                                              training=training, reuse=reuse, name=lstm_type+"_1")
        # lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, num_dirs=num_dirs,
        #                                       # use_in_bn=True,
        #                                       # use_in_drop=True, drop_rate=0.3,
        #                                       training=training, reuse=reuse, name=lstm_type+"_2")

        lstm_2 = lstm_1

        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2,
                                               # use_in_bn=True,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, time_len, 2]
    return logits, predictions


def lstm_model_v0(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v0(input_sequence, 1, params, training, reuse, name)
    return logits, predictions


def blstm_model_v0(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v0(input_sequence, 2, params, training, reuse, name)
    return logits, predictions

'''
def lstm_base_model_v0_cnn(
        input_sequence,
        num_dirs,
        params,
        training,
        reuse,
        name):
    # Very simple model
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):

        cwt_sequence = subnets_ops.cwt_local_stride_layer(
            input_sequence, params, name="cwt", use_out_bn=params["cwt_bn"], training=training,
            use_avg_pool=False, log_transform=params["log_transform"], stride_reduction_factor=1/4)

        # Some convolutions, input shape [batch, time_len, n_scales, n_spectrograms]
        out_cnn = tf.layers.conv2d(inputs=cwt_sequence, filters=16, kernel_size=3, activation=tf.nn.relu, stride=(2, 1),
                                   padding="same", name="conv_1", reuse=reuse)
        out_cnn = tf.layers.conv2d(inputs=out_cnn, filters=16, kernel_size=3, activation=tf.nn.relu, stride=(2, 1),
                                   padding="same", name="conv_2", reuse=reuse)

        # Prepare for LSTM
        temp_sequence = subnets_ops.sequence_flatten_layer(out_cnn, name="flatten")
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=params["lstm_units"], num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type+"_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=params["lstm_units"], num_dirs=num_dirs,
                                              training=training, reuse=reuse, name=lstm_type+"_2")

        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, time_len, 2]
    return logits, predictions


def lstm_model_v0_cnn(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v0_cnn(input_sequence, 1, params, training, reuse, name)
    return logits, predictions


def blstm_model_v0_cnn(
        input_sequence,
        params,
        training=False,
        reuse=False,
        name="model"):
    logits, predictions = lstm_base_model_v0_cnn(input_sequence, 2, params, training, reuse, name)
    return logits, predictions
'''


def model_for_ppt(
        input_sequence,
        params,
        training,
        reuse=False,
        name="model"):

    num_dirs = 2

    # Very simple model
    if num_dirs == 2:
        lstm_type = "blstm"
    else:
        lstm_type = "lstm"

    with tf.variable_scope(name):

        cwt_sequence = subnets_ops.cwt_local_stride_layer(
            input_sequence, params, name="cwt", use_out_bn=False, training=training,
            use_avg_pool=True, log_transform=True)

        # cwt_sequence = subnets_ops.cwt_local_stride_layer(
        #     input_sequence, params, name="cwt", use_out_bn=False, training=training,
        #     use_avg_pool=False, log_transform=True, stride_reduction_factor=1/4)

        # Convolutional stage input shape [batch, time_len, n_scales, n_spectrograms] (goal: subsampling)
        # out_cnn = subnets_ops.conv_layer(inputs=cwt_sequence, filters=16, strides=2, activation=tf.nn.relu,
        #                                  use_in_bn=True,
        #                                  training=training, reuse=reuse, name="conv_1")
        # out_cnn = subnets_ops.conv_layer(inputs=out_cnn, filters=16, strides=2, activation=tf.nn.relu,
        #                                  use_in_bn=True,
        #                                  training=training, reuse=reuse, name="conv_2")
        out_cnn = cwt_sequence

        # Flattening
        temp_sequence = subnets_ops.sequence_flatten_layer(out_cnn, name="flatten")

        with tf.name_scope("short_context"):
            in_sh = tf.shape(temp_sequence)
            zero_pad_time = tf.zeros([in_sh[0], 1, in_sh[2]], tf.float32)
            past_values = tf.concat([zero_pad_time, temp_sequence[:, :-1, :]], 1)
            future_values = tf.concat([temp_sequence[:, 1:, :], zero_pad_time], 1)
            temp_sequence = tf.concat([past_values, temp_sequence, future_values], 2)

        # Prepare for LSTM
        temp_sequence = subnets_ops.do_time_major_layer(temp_sequence, name="do_time_major")

        lstm_1 = subnets_ops.cudnn_lstm_layer(temp_sequence, num_units=256, num_dirs=num_dirs,
                                              use_in_bn=True,
                                              # use_in_drop=True, drop_rate=0.5,
                                              training=training, reuse=reuse, name=lstm_type + "_1")
        lstm_2 = subnets_ops.cudnn_lstm_layer(lstm_1, num_units=256, num_dirs=num_dirs,
                                              # use_in_bn=True,
                                              use_in_drop=True, drop_rate=0.5,
                                              training=training, reuse=reuse, name=lstm_type+"_2")

        temp_sequence = subnets_ops.undo_time_major_layer(lstm_2, name="undo_time_major")

        # logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2,
        #                                        use_in_bn=True,
        #                                        training=training, reuse=reuse, name="fc")
        logits = subnets_ops.sequence_fc_layer(temp_sequence, num_units=2,
                                               # use_in_bn=True,
                                               # use_in_drop=True, drop_rate=0.5,
                                               kernel_size=1,
                                               training=training, reuse=reuse, name="fc")
        with tf.name_scope("predictions"):
            predictions = tf.nn.softmax(logits, axis=-1)
    # Need shape [batch, time_len, 2]
    return logits, predictions
