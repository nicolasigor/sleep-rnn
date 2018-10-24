from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from spectrum.cmorlet import compute_cwt
from .constants import CHANNELS_LAST, CHANNELS_FIRST
from .constants import PAD_SAME, PAD_VALID
from .constants import BN, BN_RENORM
from .constants import MAXPOOL, AVGPOOL
from .constants import REGULAR_DROPOUT, TIME_DROPOUT


def bn_layer(
        inputs,
        name,
        type_bn=BN_RENORM,
        data_format=CHANNELS_LAST,
        reuse=False,
        training=False):
    """ type_bn should be BN or BN_RENORM
    """
    # Input sequence has shape [batch_size, ..., channels] or [batch_size, channels, ...]
    if data_format == CHANNELS_LAST:
        axis = -1
    elif data_format == CHANNELS_FIRST:
        axis = 1
    else:
        raise ValueError("Wrong data format, expected %s or %s, provided %s" %
                         (CHANNELS_FIRST, CHANNELS_LAST, data_format))
    if type_bn == BN:
        outputs = tf.layers.batch_normalization(
            inputs=inputs, training=training, name=name, reuse=reuse, renorm=False, axis=axis)
    elif type_bn == BN_RENORM:
        outputs = tf.layers.batch_normalization(
            inputs=inputs, training=training, name='%s_renorm' % name, reuse=reuse, renorm=True, axis=axis)
    else:
        raise ValueError("Wrong batchnorm value, expected '%s' or '%s', provided '%s'"
                         % (BN, BN_RENORM, type_bn))
    return outputs

# TODO: fix dropout
def dropout_layer(
        inputs,
        name,
        drop_rate=0.5,
        type_drop=TIME_DROPOUT,
        time_major=False,
        reuse=False,
        training=False):
    """ type_drop should be REGULAR_DROPOUT or TIME_DROPOUT
    """
    in_sh = tf.shape(inputs)
    if time_major:
        # Input has shape [time_len, batch, feats]
        noise_shape = [1, in_sh[1], in_sh[2], in_sh[3]]

    else:
        # Input has shape [batch, time_len, feats]
        noise_shape = [in_sh[0], 1, in_sh[2], in_sh[3]]


    # inputs = tf.layers.dropout(inputs, training=training, rate=drop_rate, name="drop")


    # noise_shape = tf.print(noise_shape, [noise_shape])
    inputs = tf.layers.dropout(inputs, training=training, rate=drop_rate,
                               name="drop", noise_shape=noise_shape)

    return inputs


def cmorlet_layer(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        stride,
        border_crop=0,
        use_avg_pool=True,
        use_log_transform=True,
        out_bn=None,
        training=False,
        data_format=CHANNELS_LAST,
        trainable_wavelet=False,
        reuse=False,
        name=None):
    """ out_bn should be BN or BN_RENORM, or None to disable
    """
    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        if use_avg_pool:
            cwt = compute_cwt(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                flattening=True, border_crop=border_crop, stride=1,
                data_format=data_format, trainable=trainable_wavelet)
            cwt = tf.layers.average_pooling2d(
                inputs=cwt, pool_size=(stride, 1), strides=(stride, 1), data_format=data_format)
        else:
            cwt = compute_cwt(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                flattening=True, border_crop=border_crop, stride=stride,
                data_format=data_format, trainable=trainable_wavelet)
        if use_log_transform:
            cwt = tf.log(cwt + 1e-3)
        if out_bn:
            cwt = bn_layer(cwt, 'bn', type_bn=out_bn, data_format=data_format, reuse=reuse, training=training)
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
    return cwt


# TODO: implement spline layer
def spline_layer():
    pass


def conv2d_layer(
        inputs,
        filters,
        kernel_size=3,
        padding=PAD_SAME,
        strides=1,
        in_bn=None,
        activation=None,
        pool=AVGPOOL,
        training=False,
        reuse=False,
        data_format=CHANNELS_LAST,
        name=None):
    """ in_bn should be BN or BN_RENORM, or None to disable
    """
    # Input sequence has shape [batch_size, height, width, channels]
    with tf.variable_scope(name):
        if in_bn:
            inputs = bn_layer(inputs, 'bn', type_bn=in_bn, data_format=data_format, reuse=reuse, training=training)

        if padding not in [PAD_SAME, PAD_VALID]:
            raise ValueError("Wrong padding, expected '%s' or '%s', provided '%s'" %
                             (PAD_VALID, PAD_SAME, padding))
        outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding, strides=strides, name='conv', reuse=reuse, data_format=data_format)

        if pool:
            if pool == AVGPOOL:
                outputs = tf.layers.average_pooling2d(inputs=outputs, pool_size=2, strides=2, data_format=data_format)
            elif pool == MAXPOOL:
                outputs = tf.layers.max_pooling2d(inputs=outputs, pool_size=2, strides=2, data_format=data_format)
            else:
                raise ValueError("Wrong value for pool, expected '%s' or '%s', provided '%s'" %
                                 (AVGPOOL, MAXPOOL, pool))
    # Ouput sequence has shape [batch_size, height, width, channels]
    return outputs


def bn_conv3x3_block(
        inputs,
        filters,
        training=False,
        reuse=False,
        name=None):
    # BN-CONV-ReLU-BN-CONV-ReLU-MaxPool block with given number of filters. Kernel of 3x3
    # with tf.variable_scope(name):
    #     outputs = conv2d_layer(inputs, filters, 3, use_in_bn=True, activation=tf.nn.relu,
    #                          training=training, reuse=reuse, name="conv3x3_1")
    #     outputs = conv2d_layer(outputs, filters, 3, use_in_bn=True, activation=tf.nn.relu, use_maxpool=True,
    #                         training=training, reuse=reuse, name="conv3x3_2")
    #return outputs
    pass


def flatten(inputs, name=None):
    with tf.name_scope(name):
        # Input has shape [batch_size, d0, ..., dn]
        dim = np.prod(inputs.get_shape().as_list()[1:])
        outputs = tf.reshape(inputs, shape=(-1, dim))
        # Output has shape [batch_size, d0*...*dn]
    return outputs


def sequence_flatten(inputs, name=None):
    with tf.name_scope(name):
        # Input has shape [batch_size, time_len, d0, ..., dn]
        dims = inputs.get_shape().as_list()
        outputs = tf.reshape(inputs, shape=(-1, dims[1], np.prod(dims[2:])))
        # Output has shape [batch_size, time_len, d0*...*dn]
    return outputs


def swap_batch_time(inputs, name=None):
    with tf.name_scope(name):
        # Input shape: [batch_size, time_len, feats] or [time_len, batch_size, feats]
        outputs = tf.transpose(inputs, (1, 0, 2))
        # Output shape: [time_len, batch_size, feats] or [batch_size, time_len, feats]
    return outputs


# TODO: fix dropout
def sequence_fc_layer(
        inputs,
        num_units,
        in_bn=None,
        in_drop=None,
        drop_rate=0,
        activation=None,
        training=False,
        reuse=False,
        name=None):
    """ in_bn should be BN or BN_RENORM, or None to disable.
    in_drop should be REGULAR_DROPOUT or TIME_DROPOUT, or None to disable
    """
    # Input sequence has shape [batch_size, time_len, feats]
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=2)  # shape [batch_size, time_len, 1, feats]
        if in_bn:
            inputs = bn_layer(inputs, 'bn', type_bn=in_bn, reuse=reuse, training=training)
        if in_drop:
            inputs = dropout_layer(inputs, 'drop', drop_rate=drop_rate, type_drop=in_drop, reuse=reuse, training=training)

        outputs = tf.layers.conv2d(inputs=inputs, filters=num_units, kernel_size=1, activation=activation,
                                   padding=PAD_SAME, name="conv1x1", reuse=reuse)

        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    # Output sequence has shape [batch_size, time_len, num_units]
    return outputs

# TODO: fix dropout and batch norm
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
