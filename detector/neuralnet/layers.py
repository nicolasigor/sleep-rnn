"""layers.py: Module that defines several useful layers for neural network models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from spectrum import cmorlet
from spectrum import spline

from utils.constants import CHANNELS_LAST, CHANNELS_FIRST
from utils.constants import PAD_SAME, PAD_VALID
from utils.constants import BN, BN_RENORM
from utils.constants import MAXPOOL, AVGPOOL
from utils.constants import SEQUENCE_DROP, REGULAR_DROP


def batchnorm_layer(
        inputs,
        name,
        batchnorm=BN,
        data_format=CHANNELS_LAST,
        reuse=False,
        training=False):
    """Buils a batchnormalization layer.

    Args:
        inputs: (tensor) Input tensor of shape [batch_size, ..., channels] or [batch_size, channels, ...]
        name: (string) A name for the operation
        batchnorm: (Optional, string, defaults to BN) Type of BN, can be BN or BN_RENORM
        data_format: (Optional, string, defaults to CHANNELS_LAST) data format used in inputs, can be CHANNELS_LAST
            or CHANNELS_FIRST.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer variables.
        training: (Optional, boolean, defaults to False) Indicates if it is the training phase or not
    """
    if data_format == CHANNELS_LAST:
        axis = -1
    elif data_format == CHANNELS_FIRST:
        axis = 1
    else:
        raise ValueError("Wrong data format, expected %s or %s, provided %s" %
                         (CHANNELS_FIRST, CHANNELS_LAST, data_format))
    if batchnorm == BN:
        outputs = tf.layers.batch_normalization(
            inputs=inputs, training=training, name=name, reuse=reuse, axis=axis)
    elif batchnorm == BN_RENORM:
        outputs = tf.layers.batch_normalization(
            inputs=inputs, training=training, name='%s_renorm' % name, reuse=reuse, renorm=True, axis=axis)
    else:
        raise ValueError("Wrong batchnorm value, expected '%s' or '%s', provided '%s'"
                         % (BN, BN_RENORM, batchnorm))
    return outputs


def dropout_layer(
        inputs,
        name,
        dropout=REGULAR_DROP,
        drop_rate=0.5,
        time_major=False,
        training=False):
    """Builds a dropout layer.

    Args:
        inputs: (3d tensor) Input tensor of shape [time_len, batch_size, n_feats] or [batch_size, time_len, n_feats]
        name: (string) A name for the operation
        dropout: (Optional, string, defaults to REGULAR_DROP) Type of dropout, can be REGULAR_DROP or SEQUENCE_DROP
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of units to be dropped.
        time_major: (Optional, boolean, defaults to False) Indicates if input is time major instead of batch major.
        training: (Optional, boolean, defaults to False) Indicates if it is the training phase or not
    """
    if dropout == SEQUENCE_DROP:
        in_shape = tf.shape(inputs)
        if time_major:  # Input has shape [time_len, batch, feats]
            noise_shape = [1, in_shape[1], in_shape[2], in_shape[3]]
        else:  # Input has shape [batch, time_len, feats]
            noise_shape = [in_shape[0], 1, in_shape[2], in_shape[3]]
        outputs = tf.layers.dropout(
            inputs, training=training, rate=drop_rate, name=name, noise_shape=noise_shape)
    elif dropout == REGULAR_DROP:
        outputs = tf.layers.dropout(
            inputs, training=training, rate=drop_rate, name=name)
    else:
        raise ValueError("Wrong dropout value, expected '%s' or '%s', provided '%s'"
                         % (SEQUENCE_DROP, REGULAR_DROP, dropout))
    return outputs


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
        use_log=True,
        batchnorm=None,
        training=False,
        data_format=CHANNELS_LAST,
        trainable_wavelet=False,
        reuse=False,
        name=None):
    """Builds the operations to compute a CWT based on the complex morlet wavelet.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        stride: (Optional, int, defaults to 1) The stride of the sliding window across the input. Default is 1.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that specifies the number of samples to be
            removed at each border at the end. This parameter allows to input a longer signal than the final desired
            size to remove border effects of the CWT.
        use_avg_pool: (Optional, boolean, defaults to True) Whether to compute the CWT with stride 1 and then compute
            an average pooling in the time axis with the given stride.
        use_log: (Optional, boolean, defaults to True) whether to apply logarithm to the CWT output (after the avg pool
            if applicable).
        batchnorm: (Optional, string, defaults to None) Type of batchnorm, can be BN or BN_RENORM. If None, batchnorm
            is not applied. The batchnorm layer is applied after the transformations.
        training: (Optional, boolean, defaults to False) Indicates if it is the training phase or not.
        data_format: (Optional, string, defaults to CHANNELS_LAST) A string from: CHANNELS_LAST, CHANNELS_FIRST.
            Specify the data format of the output data. With the default format CHANNELS_LAST, the output has shape
            [batch, signal_size, n_scales, channels]. Alternatively, with the format CHANNELS_FIRST, the output has
            shape [batch, channels, signal_size, n_scales].
        trainable_wavelet: (Optional, boolean, defaults to False) If True, the fb params will be trained with backprop.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        if use_avg_pool:
            cwt = cmorlet.compute_cwt(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                flattening=True, border_crop=border_crop, stride=1,
                data_format=data_format, trainable=trainable_wavelet)
            cwt = tf.layers.average_pooling2d(
                inputs=cwt, pool_size=(stride, 1), strides=(stride, 1), data_format=data_format)
        else:
            cwt = cmorlet.compute_cwt(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                flattening=True, border_crop=border_crop, stride=stride,
                data_format=data_format, trainable=trainable_wavelet)
        if use_log:
            cwt = tf.log(cwt + 1e-3)
        if batchnorm:
            cwt = batchnorm_layer(cwt, 'bn', batchnorm=batchnorm, data_format=data_format, reuse=reuse, training=training)
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
        batchnorm=None,
        activation=None,
        pooling=None,
        training=False,
        reuse=False,
        data_format=CHANNELS_LAST,
        name=None):
    """Buils a 2d convolutional layer with batch normalization and pooling.

    Args:
         inputs: (4d tensor) input tensor of shape [batch_size, height, width, n_channels] or
            [batch_size, n_channels, height, width]
         filters: (int) Number of filters to apply.
         kernel_size: (Optional, int or tuple of int, defaults to 3) Size of the kernels.
         padding: (Optional, string, defaults to PAD_SAME) Type of padding. Can be PAD_SAME or PAD_VALID.
         strides: (Optional, int or tuple of int, defaults to 1) Size of the strides of the convolutions.
         batchnorm: (Optional, string, defaults to None) Type of batchnorm, can be BN or BN_RENORM. If None, batchnorm
            is not applied. The batchnorm layer is applied before convolution.
         activation: (Optional, function, defaults to None) Type of activation to be used after convolution.
         pooling: (Optional, string, defaults to None) Type of pooling, can be AVGPOOL or MAXPOOL, which are always
            of stride 2 and pool size 2. If None, pooling is not applied.
         training: (Optional, boolean, defaults to False) Indicates if it is the training phase or not
         reuse: (Optional, boolean, defaults to False) Whether to reuse the layer variables.
         data_format: (Optional, string, defaults to CHANNELS_LAST) A string from: CHANNELS_LAST, CHANNELS_FIRST.
            Specify the data format of the inputs. With the default format CHANNELS_LAST, the data has shape
            [batch_size, height, width, n_channels]. Alternatively, with the format CHANNELS_FIRST, the output has
            shape [batch_size, n_channels, height, width].
         name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        if batchnorm:
            inputs = batchnorm_layer(inputs, 'bn', batchnorm=batchnorm, data_format=data_format, reuse=reuse, training=training)

        if padding not in [PAD_SAME, PAD_VALID]:
            raise ValueError("Wrong padding, expected '%s' or '%s', provided '%s'" %
                             (PAD_VALID, PAD_SAME, padding))
        outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding, strides=strides, name='conv', reuse=reuse, data_format=data_format)

        if pooling:
            if pooling == AVGPOOL:
                outputs = tf.layers.average_pooling2d(inputs=outputs, pool_size=2, strides=2, data_format=data_format)
            elif pooling == MAXPOOL:
                outputs = tf.layers.max_pooling2d(inputs=outputs, pool_size=2, strides=2, data_format=data_format)
            else:
                raise ValueError("Wrong value for pooling, expected '%s' or '%s', provided '%s'" %
                                 (AVGPOOL, MAXPOOL, pooling))
    return outputs


def bn_conv3_block(
        inputs,
        filters,
        batchnorm=BN,
        pooling=MAXPOOL,
        training=False,
        reuse=False,
        data_format=CHANNELS_LAST,
        name=None):
    """Builds a block of BN-CONV-ReLU-BN-CONV-ReLU-POOL with 3x3 kernels and same number of filters."""
    with tf.variable_scope(name):
        outputs = conv2d_layer(
            inputs, filters, batchnorm=batchnorm, activation=tf.nn.relu,
            training=training, reuse=reuse, data_format=data_format, name='conv3_1')
        outputs = conv2d_layer(
            outputs, filters, batchnorm=batchnorm, activation=tf.nn.relu, pooling=pooling,
            training=training, reuse=reuse, data_format=data_format, name='conv3_2')
    return outputs


def flatten(inputs, name=None):
    """ Flattens [batch_size, d0, ..., dn] to [batch_size, d0*...*dn]"""
    with tf.name_scope(name):
        dim = tf.reduce_prod(tf.shape(inputs)[1:])
        outputs = tf.reshape(inputs, shape=(-1, dim))
    return outputs


def sequence_flatten(inputs, name=None):
    """ Flattens [batch_size, time_len, d0, ..., dn] to [batch_size, time_len, d0*...*dn]"""
    with tf.name_scope(name):
        dims = tf.shape(inputs)
        outputs = tf.reshape(inputs, shape=(-1, dims[1], tf.reduce_prod(dims[2:])))
    return outputs


def swap_batch_time(inputs, name=None):
    """Interchange batch axis with time axis, which are assumed to be on the first and second axis."""
    with tf.name_scope(name):
        # Input shape: [batch_size, time_len, feats] or [time_len, batch_size, feats]
        outputs = tf.transpose(inputs, (1, 0, 2))
        # Output shape: [time_len, batch_size, feats] or [batch_size, time_len, feats]
    return outputs


def sequence_fc_layer(
        inputs,
        num_units,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        activation=None,
        training=False,
        reuse=False,
        name=None):
    """ Builds a FC layer that can be applied directly to a sequence, each time-step separately with the same weights.

    Args:
        inputs: (3d tensor) input tensor of shape [batch_size, time_len, n_feats]
        num_units: (int) Number of neurons for the FC layer.
        batchnorm: (Optional, string, defaults to None) Type of batchnorm, can be BN or BN_RENORM. If None, batchnorm
            is not applied. The batchnorm layer is applied before the fc layer.
        dropout: (Optional, string, defaults to None) Type of dropout, can be REGULAR_DROP or SEQUENCE_DROP. If None,
            dropout is not applied. The dropout layer is applied before the fc layer and after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of units to be dropped.
        activation: (Optional, function, defaults to None) Type of activation to be used after convolution.
        training: (Optional, boolean, defaults to False) Indicates if it is the training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=2)  # shape [batch_size, time_len, 1, feats]
        if batchnorm:
            inputs = batchnorm_layer(inputs, 'bn', batchnorm=batchnorm, reuse=reuse, training=training)
        if dropout:
            inputs = dropout_layer(inputs, 'drop', drop_rate=drop_rate, dropout=dropout, training=training)

        outputs = tf.layers.conv2d(
            inputs=inputs, filters=num_units, kernel_size=1, activation=activation, padding=PAD_SAME,
            name="conv1x1", reuse=reuse)

        outputs = tf.squeeze(outputs, axis=2, name="squeeze")  # Output has shape [batch_size, time_len, num_units]
    return outputs


def lstm_layer(
        inputs,
        num_units,
        num_dirs,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        activation=None,
        training=False,
        reuse=False,
        name=None):

    # TODO: implement lstm layer. Ask wheter there is a gpu available. If yes, cuda, if no, lstmblockfused.
    # Do inside this function the swap of batch and time dimensions. That is, inputs is batch major always.
    pass


# #
# def cudnn_lstm_layer(
#         inputs,
#                      num_units,
#                      num_dirs=1,
#                      use_in_bn=False,
#                      use_in_drop=False,
#                      drop_rate=0,
#                      training=False,
#                      reuse=False,
#                      name=None):
#     # Input_sequence has shape [time_len, batch_size, feats]
#     if num_dirs not in [1, 2]:
#         raise Exception("Expected 1 or 2 for 'num_dir'")
#
#     with tf.variable_scope(name):
#         if use_in_bn:
#             inputs = tf.layers.batch_normalization(inputs=inputs, training=training,
#                                                    name="bn", reuse=reuse, renorm=True)
#         if use_in_drop:  # Dropout mask is the same across time steps
#             noise_shape = tf.concat([[1], tf.shape(inputs)[1:]], axis=0)
#             # noise_shape = tf.print(noise_shape, [noise_shape])
#             inputs = tf.layers.dropout(inputs, training=training, rate=drop_rate,
#                                        name="drop", noise_shape=noise_shape)
#         if num_dirs == 2:
#             direction = 'bidirectional'
#             name = 'blstm'
#         else:
#             direction = 'unidirectional'
#             name = 'lstm'
#
#         rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
#                                                   num_units=num_units,
#                                                   direction=direction,
#                                                   name=name)
#
#         outputs, _ = rnn_cell(inputs)
#     # Output_sequence has shape [time_len, batch_size, num_dirs*feats]
#     return outputs
