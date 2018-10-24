from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

from models import cwt_ops


# def cwt_time_stride_layer(input_sequence,
#                           params,
#                           use_out_bn=False,
#                           training=False,
#                           reuse=False,
#                           name=None):
#     # Input sequence has shape [batch_size, time_len]
#     wavelets, _ = cwt_ops.complex_morlet_wavelets(
#         fb_array=params["fb_array"],
#         fs=params["fs"],
#         lower_freq=params["lower_freq"],
#         upper_freq=params["upper_freq"],
#         n_scales=params["n_scales"],
#         flattening=True)
#     cwt_sequence = cwt_ops.cwt_layer(
#         inputs=input_sequence,
#         wavelets=wavelets,
#         border_crop=params["border_size"],
#         stride=params["time_stride"],
#         name=name)
#     if use_out_bn:
#         cwt_sequence = tf.layers.batch_normalization(inputs=cwt_sequence, training=training,
#                                                      name=name+"bn", reuse=reuse)
#     # Output sequence has shape [batch_size, time_len, n_scales, channels]
#     return cwt_sequence


def cwt_local_stride_layer(input_sequence,
                           params,
                           stride_reduction_factor=1,
                           use_out_bn=False,
                           log_transform=False,
                           use_avg_pool=False,
                           training=False,
                           reuse=False,
                           name=None):
    # Input sequence has shape [batch_size, time_len]
    wavelets, _ = cwt_ops.complex_morlet_wavelets(
        fb_array=params["fb_array"],
        fs=params["fs"],
        lower_freq=params["lower_freq"],
        upper_freq=params["upper_freq"],
        n_scales=params["n_scales"],
        flattening=True)

    if use_avg_pool:
        cwt_sequence = cwt_ops.cwt_layer(
            inputs=input_sequence,
            wavelets=wavelets,
            border_crop=params["border_size"],
            stride=1,
            name=name)
        stride = int(params["time_stride"] * stride_reduction_factor)
        cwt_sequence = tf.layers.average_pooling2d(inputs=cwt_sequence, pool_size=(stride, 1), strides=(stride, 1))
    else:
        cwt_sequence = cwt_ops.cwt_layer(
            inputs=input_sequence,
            wavelets=wavelets,
            border_crop=params["border_size"],
            stride=int(params["time_stride"]*stride_reduction_factor),
            name=name)
    if log_transform:
        cwt_sequence = tf.log(cwt_sequence + 1e-3)
    if use_out_bn:
        cwt_sequence = tf.layers.batch_normalization(inputs=cwt_sequence, training=training,
                                                     name=name+"bn", reuse=reuse, renorm=True)
    # Output sequence has shape [batch_size, time_len, n_scales, channels]
    return cwt_sequence


def conv_layer(inputs,
               filters,
               kernel_size=3,
               padding="same",
               strides=1,
               use_in_bn=False,
               activation=None,
               use_maxpool=False,
               training=False,
               reuse=False,
               name=None):
    # Input sequence has shape [batch_size, height, width, channels]
    with tf.variable_scope(name):
        if use_in_bn:
            inputs = tf.layers.batch_normalization(inputs=inputs, training=training,
                                                   name="bn", reuse=reuse, renorm=True)

        outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding, strides=strides, name="conv", reuse=reuse)

        if use_maxpool:
            outputs = tf.layers.max_pooling2d(inputs=outputs, pool_size=2, strides=2)
    # Ouput sequence has shape [batch_size, height, width, channels]
    return outputs


# def flatten_layer(inputs, name=None):
#     with tf.name_scope(name):
#         # Input has shape [batch_size, d0, ..., dn]
#         dim = np.prod(inputs.get_shape().as_list()[1:])
#         outputs = tf.reshape(inputs, shape=(-1, dim))
#         # Output has shape [batch_size, d0*...*dn]
#     return outputs


def sequence_flatten_layer(inputs, name=None):
    with tf.name_scope(name):
        # Input has shape [batch_size, time_len, d0, ..., dn]
        dims = inputs.get_shape().as_list()
        outputs = tf.reshape(inputs, shape=(-1, dims[1], np.prod(dims[2:])))
        # Output has shape [batch_size, time_len, d0*...*dn]
    return outputs


def do_time_major_layer(inputs, name=None):
    with tf.name_scope(name):
        # Input shape: [batch_size, time_len, feats]
        outputs = tf.transpose(inputs, (1, 0, 2))
        # Output shape: [time_len, batch_size, feats]
    return outputs


def cudnn_lstm_layer(inputs,
                     num_units,
                     num_dirs=1,
                     use_in_bn=False,
                     use_in_drop=False,
                     drop_rate=0,
                     training=False,
                     reuse=False,
                     name=None):
    # Input_sequence has shape [time_len, batch_size, feats]
    if num_dirs not in [1, 2]:
        raise Exception("Expected 1 or 2 for 'num_dir'")

    with tf.variable_scope(name):
        if use_in_bn:
            inputs = tf.layers.batch_normalization(inputs=inputs, training=training,
                                                   name="bn", reuse=reuse, renorm=True)
        if use_in_drop:  # Dropout mask is the same across time steps
            noise_shape = tf.concat([[1], tf.shape(inputs)[1:]], axis=0)
            # noise_shape = tf.print(noise_shape, [noise_shape])
            inputs = tf.layers.dropout(inputs, training=training, rate=drop_rate,
                                       name="drop", noise_shape=noise_shape)
        if num_dirs == 2:
            direction = 'bidirectional'
            name = 'blstm'
        else:
            direction = 'unidirectional'
            name = 'lstm'

        rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                  num_units=num_units,
                                                  direction=direction,
                                                  name=name)

        outputs, _ = rnn_cell(inputs)
    # Output_sequence has shape [time_len, batch_size, num_dirs*feats]
    return outputs







def undo_time_major_layer(inputs, name=None):
    with tf.name_scope(name):
        # Input shape: [time_len, batch_size, feats]
        outputs = tf.transpose(inputs, (1, 0, 2))
        # Output shape: [batch_size, time_len, feats]
    return outputs


def sequence_fc_layer(inputs,
                      num_units,
                      use_in_bn=False,
                      use_in_drop=False,
                      kernel_size=1,
                      drop_rate=0,
                      activation=None,
                      training=False,
                      reuse=False,
                      name=None):
    # Input sequence has shape [batch_size, time_len, feats]
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=2)  # shape [batch_size, time_len, 1, feats]
        if use_in_bn:
            inputs = tf.layers.batch_normalization(inputs=inputs, training=training, name="bn", reuse=reuse, renorm=True)
        if use_in_drop:
            # inputs = tf.layers.dropout(inputs, training=training, rate=drop_rate, name="drop")
            in_sh = tf.shape(inputs)
            noise_shape = [in_sh[0], 1, in_sh[2], in_sh[3]]
            # noise_shape = tf.print(noise_shape, [noise_shape])
            inputs = tf.layers.dropout(inputs, training=training, rate=drop_rate,
                                       name="drop", noise_shape=noise_shape)

        outputs = tf.layers.conv2d(inputs=inputs, filters=num_units, kernel_size=kernel_size, activation=activation,
                                   padding="same", name="conv1x1", reuse=reuse)

        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    # Output sequence has shape [batch_size, time_len, num_units]
    return outputs


def bn_conv3x3_block(inputs,
                     filters,
                     training=False,
                     reuse=False,
                     name=None):
    # BN-CONV-ReLU-BN-CONV-ReLU-MaxPool block with given number of filters. Kernel of 3x3
    with tf.variable_scope(name):
        outputs = conv_layer(inputs, filters, 3, use_in_bn=True, activation=tf.nn.relu,
                             training=training, reuse=reuse, name="conv3x3_1")
        outputs = conv_layer(outputs, filters, 3, use_in_bn=True, activation=tf.nn.relu, use_maxpool=True,
                             training=training, reuse=reuse, name="conv3x3_2")
    return outputs
