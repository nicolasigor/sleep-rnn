"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers

from utils import constants
from utils import errors
from utils import param_keys


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
    errors.check_valid_value(
        params[param_keys.TYPE_WAVELET], param_keys.TYPE_WAVELET,
        [constants.CMORLET, constants.SPLINE])

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
        if params[param_keys.TYPE_WAVELET] == constants.CMORLET:
            outputs = layers.cmorlet_layer(
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
        else:  # For now we do the same
            # TODO: use here the spline layer.
            raise NotImplementedError(
                'Type spline for wavelet not implemented.')

        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[param_keys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            batchnorm_first_lstm=params[param_keys.TYPE_BATCHNORM],
            dropout_rest_lstm=params[param_keys.TYPE_DROPOUT],
            drop_rate=params[param_keys.DROP_RATE_HIDDEN],
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
        return logits, probabilities


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
    errors.check_valid_value(
        params[param_keys.TYPE_WAVELET], param_keys.TYPE_WAVELET,
        [constants.CMORLET, constants.SPLINE])

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
        if params[param_keys.TYPE_WAVELET] == constants.CMORLET:
            outputs = layers.cmorlet_layer(
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
        else:
            raise NotImplementedError(
                'Type spline for wavelet not implemented.')

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
            drop_rate=params[param_keys.DROP_RATE_HIDDEN],
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
        return logits, probabilities


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
    errors.check_valid_value(
        params[param_keys.TYPE_WAVELET], param_keys.TYPE_WAVELET,
        [constants.CMORLET, constants.SPLINE])

    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[param_keys.BORDER_DURATION] * params[param_keys.FS])
        if params[param_keys.TYPE_WAVELET] == constants.CMORLET:
            outputs = layers.cmorlet_layer(
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
        else:
            raise NotImplementedError(
                'Type spline for wavelet not implemented.')

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
            drop_rate=params[param_keys.DROP_RATE_HIDDEN],
            training=training,
            name='multi_layer_blstm')

        if params[param_keys.FC_UNITS] > 0:
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
                kernel_init=tf.initializers.he_normal(),
                dropout=params[param_keys.TYPE_DROPOUT],
                drop_rate=params[param_keys.DROP_RATE_OUTPUT],
                training=training,
                name='logits')
        else:
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
        return logits, probabilities


#
# def wavelet_blstm_net(
#         inputs,
#         params,
#         training,
#         name='model'):
#     """ Wavelet transform, convolutions, and BLSTM to make a prediction.
#
#     This models first computes the CWT to form scalograms, then, if
#     applicable, applies convolutional blocks of two 3x3 convolutions followed
#     by pooling. After this, the outputs is flatten and is passed to a
#     blstm stage. This stage is a 2-layers BLSTM if only one time level is used,
#     or it's a ladder network downsampling and upsampling the time dimension if
#     two or three time levels are used. The final classification is made with a
#     FC layer with 2 outputs.
#
#     Args:
#         inputs: (2d tensor) input tensor of shape [batch_size, time_len]
#         params: (dict) Parameters to configure the model (see utils.model_keys)
#         training: (boolean) Indicates if it is the training phase or not.
#         name: (Optional, string, defaults to 'model') A name for the network.
#     """
#     with tf.variable_scope(name):
#
#         errors.check_valid_value(
#             params[param_keys.N_CONV_BLOCKS],
#             param_keys.N_CONV_BLOCKS, [0, 1, 2, 3])
#         errors.check_valid_value(
#             params[param_keys.N_TIME_LEVELS],
#             param_keys.N_TIME_LEVELS, [1, 2, 3])
#         errors.check_valid_value(
#             params[param_keys.TYPE_WAVELET], param_keys.TYPE_WAVELET,
#             [constants.CMORLET, constants.SPLINE])
#
#         factor = params[param_keys.TIME_RESOLUTION_FACTOR]
#         cwt_stride = factor / (2 ** params[param_keys.N_CONV_BLOCKS])
#
#         # CWT stage
#         border_crop = int(
#             params[param_keys.BORDER_DURATION] * params[param_keys.FS])
#         if params[param_keys.TYPE_WAVELET] == constants.CMORLET:
#             outputs = layers.cmorlet_layer(
#                 inputs,
#                 params[param_keys.FB_LIST],
#                 params[param_keys.FS],
#                 lower_freq=params[param_keys.LOWER_FREQ],
#                 upper_freq=params[param_keys.UPPER_FREQ],
#                 n_scales=params[param_keys.N_SCALES],
#                 stride=cwt_stride,
#                 size_factor=params[param_keys.WAVELET_SIZE_FACTOR],
#                 border_crop=border_crop,
#                 use_log=params[param_keys.USE_LOG],
#                 training=training,
#                 trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
#                 name='spectrum')
#         else:  # For now we do the same
#             outputs = layers.cmorlet_layer(
#                 inputs,
#                 params[param_keys.FB_LIST],
#                 params[param_keys.FS],
#                 lower_freq=params[param_keys.LOWER_FREQ],
#                 upper_freq=params[param_keys.UPPER_FREQ],
#                 n_scales=params[param_keys.N_SCALES],
#                 stride=cwt_stride,
#                 border_crop=border_crop,
#                 use_log=params[param_keys.USE_LOG],
#                 training=training,
#                 trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
#                 name='spectrum')
#
#         # Convolutional stage (only if n_conv_blocks is greater than 0)
#         for i in range(params[param_keys.N_CONV_BLOCKS]):
#             filters = params[param_keys.INITIAL_CONV_FILTERS] * (2 ** i)
#             outputs = layers.bn_conv3_block(
#                 outputs,
#                 filters,
#                 batchnorm=params[param_keys.BATCHNORM_CONV],
#                 training=training,
#                 pooling=params[param_keys.POOLING_CONV],
#                 residual=params[param_keys.RESIDUAL_CONV],
#                 name='conv_block_%d' % (i+1))
#
#         outputs = layers.sequence_flatten(outputs, 'flatten')
#
#         if params[param_keys.N_TIME_LEVELS] == 1:  # Multilayer BLSTM (2 layers)
#             outputs = layers.multilayer_lstm_block(
#                 outputs,
#                 params[param_keys.INITIAL_LSTM_UNITS],
#                 2,
#                 num_dirs=constants.BIDIRECTIONAL,
#                 batchnorm_first_lstm=params[param_keys.BATCHNORM_FIRST_LSTM],
#                 dropout_first_lstm=params[param_keys.DROPOUT_FIRST_LSTM],
#                 batchnorm_rest_lstm=params[param_keys.BATCHNORM_REST_LSTM],
#                 dropout_rest_lstm=params[param_keys.DROPOUT_REST_LSTM],
#                 drop_rate=params[param_keys.DROP_RATE_LSTM],
#                 training=training,
#                 name='multi_layer_blstm')
#         else:  # Multi stage BLSTM
#             outputs = layers.multistage_lstm_block(
#                 outputs,
#                 params[param_keys.INITIAL_LSTM_UNITS],
#                 params[param_keys.N_TIME_LEVELS],
#                 duplicate_after_downsampling=params[param_keys.DUPLICATE_AFTER_DOWNSAMPLING_LSTM],
#                 num_dirs=constants.BIDIRECTIONAL,
#                 batchnorm_first_lstm=params[param_keys.BATCHNORM_FIRST_LSTM],
#                 dropout_first_lstm=params[param_keys.DROPOUT_FIRST_LSTM],
#                 batchnorm_rest_lstm=params[param_keys.BATCHNORM_REST_LSTM],
#                 dropout_rest_lstm=params[param_keys.DROPOUT_REST_LSTM],
#                 time_pooling=params[param_keys.TIME_POOLING],
#                 drop_rate=params[param_keys.DROP_RATE_LSTM],
#                 training=training,
#                 name='multi_stage_blstm')
#
#         # Final FC classification layer
#         logits = layers.sequence_fc_layer(
#             outputs,
#             2,
#             batchnorm=params[param_keys.BATCHNORM_FC],
#             dropout=params[param_keys.DROPOUT_FC],
#             drop_rate=params[param_keys.DROP_RATE_FC],
#             training=training,
#             name='fc'
#         )
#         with tf.variable_scope('probabilities'):
#             probabilities = tf.nn.softmax(logits)
#         return logits, probabilities
