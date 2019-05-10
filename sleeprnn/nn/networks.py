"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers

from sleep.common import constants
from sleep.common import checks
from sleep.common import pkeys


def dummy_net(
        inputs,
        params,
        training,
        name='model_dummy'
):
    """ Dummy network used for debugging purposes."""
    with tf.variable_scope(name):
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        inputs = inputs[:, border_crop:-border_crop]
        # Simulates downsampling by 8
        inputs = inputs[:, ::8]
        # Simulates shape [batch, time, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            inputs,
            2,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')
        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
    return logits, probabilities


def wavelet_blstm_net_v1(
        inputs,
        params,
        training,
        name='model_v1'
):
    """ Wavelet transform and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms. After this, the
    outputs is flatten and is passed to a 2-layers BLSTM.
    The final classification is made with a FC layer with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            batchnorm_first_lstm=params[pkeys.TYPE_BATCHNORM],
            dropout_first_lstm=None,
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v4(
        inputs,
        params,
        training,
        name='model_v4'
):
    """ Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v4') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn
