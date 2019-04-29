"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers

from sleep.utils import constants
from sleep.utils import checks
from sleep.utils import param_keys


def dummy_net(
        inputs,
        params,
        training,
        name='model_dummy'
):
    """ Dummy network used for debugging purposes."""
    with tf.variable_scope(name):
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
        inputs = inputs[:, border_crop:-border_crop]
        # Simulates downsampling by 8
        inputs = inputs[:, ::8]
        # Simulates shape [batch, time, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            inputs,
            2,
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
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
        params: (dict) Parameters to configure the model (see utils.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=8,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            name='spectrum')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            batchnorm_first_lstm=params[param_keys.TYPE_BATCHNORM],
            dropout_first_lstm=None,
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[param_keys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')
        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v2(
        inputs,
        params,
        training,
        name='model_v2'
):
    """ Wavelet transform, resnet, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a convolutional stage with residual units.
    . After this, the outputs is flatten and is passed to a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see utils.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')

        # Convolutional stage with residual units
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_residualv2_block(
            outputs,
            init_filters,
            training,
            is_first_unit=True,
            strides=2,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_1')
        outputs = layers.conv2d_residualv2_block(
            outputs,
            init_filters * 2,
            training,
            strides=2,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_2')
        outputs = layers.conv2d_residualv2_block(
            outputs,
            init_filters * 4,
            training,
            strides=2,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_3')
        # After last residual unit, we need to perform an additional relu
        outputs = tf.nn.relu(outputs)

        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            batchnorm_first_lstm=params[param_keys.TYPE_BATCHNORM],
            dropout_first_lstm=params[param_keys.TYPE_DROPOUT],
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[param_keys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        # Additional FC layer to increase model flexibility
        outputs = layers.sequence_fc_layer(
            outputs,
            params[param_keys.FC_UNITS],
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')
        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v3(
        inputs,
        params,
        training,
        name='model_v3'
):
    """ Wavelet transform, resnet, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a convolutional stage with residual units
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see utils.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
    if params[param_keys.TYPE_WAVELET] == constants.CMORLET:
        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')

        # Convolutional stage with residual units
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_residualv2_prebn_block(
            outputs,
            init_filters,
            training,
            is_first_unit=True,
            strides=2,
            kernel_init=tf.initializers.he_normal(),
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_1')
        outputs = layers.conv2d_residualv2_prebn_block(
            outputs,
            init_filters * 2,
            training,
            strides=2,
            kernel_init=tf.initializers.he_normal(),
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_2')
        outputs = layers.conv2d_residualv2_prebn_block(
            outputs,
            init_filters * 4,
            training,
            strides=2,
            kernel_init=tf.initializers.he_normal(),
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_3')
        # After last residual unit, we need to perform an additional BN+relu
        if params[param_keys.TYPE_BATCHNORM]:
            outputs = layers.batchnorm_layer(
                outputs, 'bn_last', batchnorm=params[param_keys.TYPE_BATCHNORM],
                training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[param_keys.TYPE_DROPOUT],
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[param_keys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[param_keys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[param_keys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v3_ff(
        inputs,
        params,
        training,
        name='model_v3_ff'
):
    """ Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see utils.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')

        # Convolutional stage (standard feed-forward)
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[param_keys.INITIAL_KERNEL_SIZE],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[param_keys.TYPE_DROPOUT],
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[param_keys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[param_keys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[param_keys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def experimental_net(
        inputs,
        params,
        training,
        name='experimental'
):

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')


        # Experimental feature
        outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[param_keys.INITIAL_KERNEL_SIZE],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[param_keys.TYPE_DROPOUT],
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[param_keys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[param_keys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[param_keys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
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

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')

        # ReLU after CWT
        outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # 1D convolutions

        # Convolutional stage (standard feed-forward)
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv1d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[param_keys.INITIAL_KERNEL_SIZE],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')
        outputs = layers.conv1d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')
        outputs = layers.conv1d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[param_keys.TYPE_DROPOUT],
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[param_keys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[param_keys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[param_keys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_conv_net_v3(
        inputs,
        params,
        training,
        name='model_v3_conv'
):
    """ Wavelet transform, resnet, and 1d conv to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a convolutional stage with residual units
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers conv1D (baseline network).
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see utils.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')


        # Convolutional stage with residual units
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_residualv2_prebn_block(
            outputs,
            init_filters,
            training,
            is_first_unit=True,
            strides=2,
            kernel_init=tf.initializers.he_normal(),
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_1')
        outputs = layers.conv2d_residualv2_prebn_block(
            outputs,
            init_filters * 2,
            training,
            strides=2,
            kernel_init=tf.initializers.he_normal(),
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_2')
        outputs = layers.conv2d_residualv2_prebn_block(
            outputs,
            init_filters * 4,
            training,
            strides=2,
            kernel_init=tf.initializers.he_normal(),
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='res_3')
        # After last residual unit, we need to perform an additional BN+relu
        if params[param_keys.TYPE_BATCHNORM]:
            outputs = layers.batchnorm_layer(
                outputs, 'bn_last', batchnorm=params[param_keys.TYPE_BATCHNORM],
                training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Two 1D layers
        with tf.variable_scope('conv1d'):
            outputs = tf.expand_dims(outputs, axis=2)

            if params[param_keys.TYPE_BATCHNORM]:
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_1' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal(),
                    use_bias=False)
                outputs = layers.batchnorm_layer(
                    outputs, 'bn_1',
                    batchnorm=params[param_keys.TYPE_BATCHNORM],
                    training=training, scale=False)
                outputs = tf.nn.relu(outputs)

                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_2' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal(),
                    use_bias=False)
                outputs = layers.batchnorm_layer(
                    outputs, 'bn_2',
                    batchnorm=params[param_keys.TYPE_BATCHNORM],
                    training=training, scale=False)
                outputs = tf.nn.relu(outputs)

            else:
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_1' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal())
                outputs = tf.nn.relu(outputs)

                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_2' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal())
                outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        if params[param_keys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[param_keys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_conv_net_v3_ff(
        inputs,
        params,
        training,
        name='model_v3_ff_conv'
):
    """ Wavelet transform, conv, and 1D conv to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers conv1D (baseline network).
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see utils.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[param_keys.FB_LIST],
            params[param_keys.FS],
            lower_freq=params[param_keys.LOWER_FREQ],
            upper_freq=params[param_keys.UPPER_FREQ],
            n_scales=params[param_keys.N_SCALES],
            stride=1,
            size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[param_keys.USE_LOG],
            training=training,
            trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            name='spectrum')

        # Convolutional stage (standard feed-forward)
        init_filters = params[param_keys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[param_keys.INITIAL_KERNEL_SIZE],
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_2')
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[param_keys.TYPE_BATCHNORM],
            downsampling=params[param_keys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_3')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Two 1D layers
        with tf.variable_scope('conv1d'):
            outputs = tf.expand_dims(outputs, axis=2)

            if params[param_keys.TYPE_BATCHNORM]:
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_1' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal(),
                    use_bias=False)
                outputs = layers.batchnorm_layer(
                    outputs, 'bn_1',
                    batchnorm=params[param_keys.TYPE_BATCHNORM],
                    training=training, scale=False)
                outputs = tf.nn.relu(outputs)

                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_2' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal(),
                    use_bias=False)
                outputs = layers.batchnorm_layer(
                    outputs, 'bn_2',
                    batchnorm=params[param_keys.TYPE_BATCHNORM],
                    training=training, scale=False)
                outputs = tf.nn.relu(outputs)

            else:
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_1' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal())
                outputs = tf.nn.relu(outputs)

                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[param_keys.CONV_1D_FILTERS],
                    kernel_size=(params[param_keys.CONV_1D_KERNEL], 1),
                    padding=constants.PAD_SAME,
                    strides=1,
                    name='conv%d_2' % params[param_keys.CONV_1D_KERNEL],
                    kernel_initializer=tf.initializers.he_normal())
                outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        if params[param_keys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[param_keys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_OUTPUT],
            training=training,
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn
