"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers
from . import spectrum

from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys


def dummy_net(
        inputs,
        params,
        training,
        name='model_dummy'
):
    """ Dummy network used for debugging purposes."""
    print('Using model DUMMY')
    with tf.variable_scope(name):
        cwt_prebn = None
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
    return logits, probabilities, cwt_prebn


def debug_net(
        inputs,
        params,
        training,
        name='model_debug'
):
    """ Dummy network used for debugging purposes."""
    print('Using DEBUG net')
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # Shape is [batch, time, n_scales, n_channels]
        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=64, #params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            16,#init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,  # init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

        # outputs = layers.conv2d_prebn_block(
        #     outputs,
        #     2,  # init_filters,
        #     training,
        #     batchnorm=params[pkeys.TYPE_BATCHNORM],
        #     downsampling=params[pkeys.CONV_DOWNSAMPLING],
        #     kernel_init=tf.initializers.he_normal(),
        #     name='convblock_3')

        # Flattening for dense part, shape [batch, time, feats]
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            128, #params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=1,
            num_dirs=constants.UNIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

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
    print('Using model V1')
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
    print('Using model V4')
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


def wavelet_blstm_net_v5(
        inputs,
        params,
        training,
        name='model_v5'
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
        name: (Optional, string, defaults to 'model_v5') A name for the network.
    """
    print('Using model V5')
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
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        # Maxpool in frequencies, then avg in time
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1))

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

        # Maxpool in frequencies, then avg in time
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1))

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # Maxpool in frequencies, then avg in time
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1))

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


def wavelet_blstm_net_v6(
        inputs,
        params,
        training,
        name='model_v6'
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
        name: (Optional, string, defaults to 'model_v6') A name for the network.
    """
    print('Using model V6')
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
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        # avg in time, then Maxpool in frequencies
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1))
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

        # avg in time, then Maxpool in frequencies
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1))
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # avg in time, then Maxpool in frequencies
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1))
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2))

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


def wavelet_blstm_net_v7(
        inputs,
        params,
        training,
        name='model_v7'
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
        name: (Optional, string, defaults to 'model_v7') A name for the network.
    """
    print('Using model V7')
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
            init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
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


def wavelet_blstm_net_v7_lite(
        inputs,
        params,
        training,
        name='model_v7_lite'
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
        name: (Optional, string, defaults to 'model_v7_lite') A name for the network.
    """
    print('Using model V7 LITE')
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
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        # outputs = tf.layers.average_pooling2d(
        #     inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

        # outputs = layers.conv2d_prebn_block(
        #     outputs,
        #     init_filters,
        #     training,
        #     batchnorm=params[pkeys.TYPE_BATCHNORM],
        #     downsampling=params[pkeys.CONV_DOWNSAMPLING],
        #     kernel_init=tf.initializers.he_normal(),
        #     name='convblock_3')

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


def wavelet_blstm_net_v7_litebig(
        inputs,
        params,
        training,
        name='model_v7_litebig'
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
        name: (Optional, string, defaults to 'model_v7_litebig') A name for the network.
    """
    print('Using model V7 LITE BIG')
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
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        # outputs = tf.layers.average_pooling2d(
        #     inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

        # outputs = layers.conv2d_prebn_block(
        #     outputs,
        #     init_filters,
        #     training,
        #     batchnorm=params[pkeys.TYPE_BATCHNORM],
        #     downsampling=params[pkeys.CONV_DOWNSAMPLING],
        #     kernel_init=tf.initializers.he_normal(),
        #     name='convblock_3')

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


def wavelet_blstm_net_v8(
        inputs,
        params,
        training,
        name='model_v8'
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
        name: (Optional, string, defaults to 'model_v8') A name for the network.
    """
    print('Using model V8')
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
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Split magnitude and phase
        outputs_magnitude = outputs[..., 0:1]
        outputs_phase = outputs[..., 1:]

        # MAGNITUDE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_magnitude = layers.conv2d_prebn_block(
            outputs_magnitude,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1_mag')
        outputs_magnitude = layers.conv2d_prebn_block(
            outputs_magnitude,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2_mag')

        # PHASE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_phase = layers.conv2d_prebn_block(
            outputs_phase,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1_pha')
        outputs_phase = layers.conv2d_prebn_block(
            outputs_phase,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2_pha')

        # Now concatenate magnitude and phase paths
        outputs = tf.concat([outputs_magnitude, outputs_phase], axis=-1)

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


def wavelet_blstm_net_v9(
        inputs,
        params,
        training,
        name='model_v9'
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
        name: (Optional, string, defaults to 'model_v9') A name for the network.
    """
    print('Using model V9')
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
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Split magnitude and phase
        outputs_magnitude = outputs[..., 0:1]
        outputs_phase = outputs[..., 1:]

        # MAGNITUDE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_magnitude = layers.conv2d_prebn_block(
            outputs_magnitude,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1_mag')

        # PHASE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_phase = layers.conv2d_prebn_block(
            outputs_phase,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1_pha')

        # Now concatenate magnitude and phase paths
        outputs = tf.concat([outputs_magnitude, outputs_phase], axis=-1)

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

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


def wavelet_blstm_net_v10(
        inputs,
        params,
        training,
        name='model_v10'
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
        name: (Optional, string, defaults to 'model_v10') A name for the network.
    """
    print('Using model V10')
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
            stride=2,
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
            init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

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


def wavelet_blstm_net_v11(
        inputs,
        params,
        training,
        name='model_v11'
):
    """ conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print('Using model V11 (Time-Domain)')
    with tf.variable_scope(name):

        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_1')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_2')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_3')

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
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v12(
        inputs,
        params,
        training,
        name='model_v12'
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
        name: (Optional, string, defaults to 'model_v12') A name for the network.
    """
    print('Using model V12 (cwt)')
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
            stride=2,
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
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

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


def wavelet_blstm_net_v13(
        inputs,
        params,
        training,
        name='model_v13'
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
        name: (Optional, string, defaults to 'model_v13') A name for the network.
    """
    print('Using model V13 (cwt using freqs as channels)')
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
            stride=2,
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

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

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


def wavelet_blstm_net_v14(
        inputs,
        params,
        training,
        name='model_v14'
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
        name: (Optional, string, defaults to 'model_v14') A name for the network.
    """
    print('Using model V14 (cwt using freqs as channels)')
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

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

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


def wavelet_blstm_net_v15(
        inputs,
        params,
        training,
        name='model_v15'
):
    """ conv1D in time, conv2D in cwt, and BLSTM to make a prediction.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v15') A name for the network.
    """
    print('Using model V15 (Time_11 + CWT_12)')
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_1')

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_2')

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_3')

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs_cwt = tf.nn.relu(outputs_cwt)

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_cwt2d_1')

        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_cwt2d_2')

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, 'flatten')

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

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
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v16(
        inputs,
        params,
        training,
        name='model_v16'
):
    """ conv1D in time, conv1D in cwt, and BLSTM to make a prediction.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v16') A name for the network.
    """
    print('Using model V16 (Time_11 + CWT_13)')
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_1')

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_2')

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_3')

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs_cwt = tf.nn.relu(outputs_cwt)

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, 'flatten')

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv1d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_cwt1d_1')

        outputs_cwt = layers.conv1d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_cwt1d_2')

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

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
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v17(
        inputs,
        params,
        training,
        name='model_v17'
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
        name: (Optional, string, defaults to 'model_v17') A name for the network.
    """
    print('Using model V17 (cwt with real and imaginary parts directly)')
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_rectangular(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
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
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')

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


def wavelet_blstm_net_v18(
        inputs,
        params,
        training,
        name='model_v18'
):
    """ conv1D in time, conv2D in cwt, and BLSTM to make a prediction.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v18') A name for the network.
    """
    print('Using model V18 (Time_11 + CWT_17)')
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_1')

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_2')

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_time1d_3')

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer_rectangular(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

        if params[pkeys.USE_RELU]:
            outputs_cwt = tf.nn.relu(outputs_cwt)

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_cwt2d_1')

        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_cwt2d_2')

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, 'flatten')

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

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
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v20_concat(
        inputs,
        params,
        training,
        name='model_v20_concat'
):
    """ conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v20_concat') A name for the network.
    """
    print('Using model V20_CONCAT (Time-Domain + sigma band)')
    with tf.variable_scope(name):
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # compute sigma band
        inputs_sigma = spectrum.compute_sigma_band(
            inputs,
            fs=params[pkeys.FS],
            ntaps=params[pkeys.SIGMA_FILTER_NTAPS],
            border_crop=border_crop)

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # ----- CONCATENATE INPUT AND SIGMA ACROSS CHANNELS
        inputs = tf.concat([inputs, inputs_sigma], axis=2)
        # -----

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_1')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_2')

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_3')

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
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v20_indep(
        inputs,
        params,
        training,
        name='model_v20_indep'
):
    """ conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v20_indep') A name for the network.
    """
    print('Using model V20_INDEP (Time-Domain + sigma band)')
    with tf.variable_scope(name):
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # compute sigma band
        inputs_sigma = spectrum.compute_sigma_band(
            inputs,
            fs=params[pkeys.FS],
            ntaps=params[pkeys.SIGMA_FILTER_NTAPS],
            border_crop=border_crop)

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_original = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_original = tf.expand_dims(inputs_original, axis=2)

        # Each channel is processed independently

        with tf.variable_scope('tower_original'):
            # BN at input
            outputs_original = layers.batchnorm_layer(
                inputs_original, 'bn_input',
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training)

            # 1D convolutions expect shape [batch, time_len, n_feats]

            # Convolutional stage (standard feed-forward)
            outputs_original = layers.conv1d_prebn_block(
                outputs_original,
                params[pkeys.TIME_CONV_FILTERS_1],
                training,
                kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_1')

            outputs_original = layers.conv1d_prebn_block(
                outputs_original,
                params[pkeys.TIME_CONV_FILTERS_2],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_2')

            outputs_original = layers.conv1d_prebn_block(
                outputs_original,
                params[pkeys.TIME_CONV_FILTERS_3],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_3')

        with tf.variable_scope('tower_sigma'):
            # BN at input
            outputs_sigma = layers.batchnorm_layer(
                inputs_sigma, 'bn_input',
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training)

            # 1D convolutions expect shape [batch, time_len, n_feats]

            # Convolutional stage (standard feed-forward)
            outputs_sigma = layers.conv1d_prebn_block(
                outputs_sigma,
                params[pkeys.TIME_CONV_FILTERS_1],
                training,
                kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_1')

            outputs_sigma = layers.conv1d_prebn_block(
                outputs_sigma,
                params[pkeys.TIME_CONV_FILTERS_2],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_2')

            outputs_sigma = layers.conv1d_prebn_block(
                outputs_sigma,
                params[pkeys.TIME_CONV_FILTERS_3],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_3')

        # Concatenate both paths
        outputs = tf.concat([outputs_original, outputs_sigma], axis=-1)

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
        cwt_prebn = None
        return logits, probabilities, cwt_prebn
