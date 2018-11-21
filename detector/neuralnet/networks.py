"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers

from utils import constants
from utils import errors
from utils import param_keys


def wavelet_blstm_net(
        inputs,
        params,
        training,
        name='model'):
    """ Wavelet transform, convolutions, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, then, if
    applicable, applies convolutional blocks of two 3x3 convolutions followed
    by pooling. After this, the outputs is flatten and is passed to a
    blstm stage. This stage is a 2-layers BLSTM if only one time level is used,
    or it's a ladder network downsampling and upsampling the time dimension if
    two or three time levels are used. The final classification is made with a
    FC layer with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see utils.model_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """
    with tf.variable_scope(name):

        errors.check_valid_value(
            params[param_keys.N_CONV_BLOCKS],
            param_keys.N_CONV_BLOCKS, [0, 1, 2, 3])
        errors.check_valid_value(
            params[param_keys.N_TIME_LEVELS],
            param_keys.N_TIME_LEVELS, [1, 2, 3])
        errors.check_valid_value(
            params[param_keys.TYPE_WAVELET], param_keys.TYPE_WAVELET,
            [constants.CMORLET, constants.SPLINE])

        factor = params[param_keys.TIME_RESOLUTION_FACTOR]
        cwt_stride = factor / (2 ** params[param_keys.N_CONV_BLOCKS])

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
                stride=cwt_stride,
                border_crop=border_crop,
                use_log=params[param_keys.USE_LOG],
                training=training,
                trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
                name='spectrum')
        else:  # For now we do the same
            # TODO: use here the spline layer.
            outputs = layers.cmorlet_layer(
                inputs,
                params[param_keys.FB_LIST],
                params[param_keys.FS],
                lower_freq=params[param_keys.LOWER_FREQ],
                upper_freq=params[param_keys.UPPER_FREQ],
                n_scales=params[param_keys.N_SCALES],
                stride=cwt_stride,
                border_crop=border_crop,
                use_log=params[param_keys.USE_LOG],
                training=training,
                trainable_wavelet=params[param_keys.TRAINABLE_WAVELET],
                name='spectrum')

        # Convolutional stage (only if n_conv_blocks is greater than 0)
        for i in range(params[param_keys.N_CONV_BLOCKS]):
            filters = params[param_keys.INITIAL_CONV_FILTERS] * (2 ** i)
            outputs = layers.bn_conv3_block(
                outputs,
                filters,
                batchnorm=params[param_keys.BATCHNORM_CONV],
                training=training,
                name='conv_block_%d' % (i+1))

        outputs = layers.sequence_flatten(outputs, 'flatten')

        if params[param_keys.N_TIME_LEVELS] == 1:  # Multilayer BLSTM (2 layers)
            outputs = layers.multilayer_lstm_block(
                outputs,
                params[param_keys.INITIAL_LSTM_UNITS],
                2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm_first_lstm=params[param_keys.BATCHNORM_FIRST_LSTM],
                dropout_first_lstm=params[param_keys.DROPOUT_FIRST_LSTM],
                batchnorm_rest_lstm=params[param_keys.BATCHNORM_REST_LSTM],
                dropout_rest_lstm=params[param_keys.DROPOUT_REST_LSTM],
                drop_rate=params[param_keys.DROP_RATE],
                training=training,
                name='multi_layer_blstm')
        else:  # Multi stage BLSTM
            outputs = layers.multistage_lstm_block(
                outputs,
                params[param_keys.INITIAL_LSTM_UNITS],
                params[param_keys.N_TIME_LEVELS],
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm_first_lstm=params[param_keys.BATCHNORM_FIRST_LSTM],
                dropout_first_lstm=params[param_keys.DROPOUT_FIRST_LSTM],
                batchnorm_rest_lstm=params[param_keys.BATCHNORM_REST_LSTM],
                dropout_rest_lstm=params[param_keys.DROPOUT_REST_LSTM],
                time_pooling=params[param_keys.TIME_POOLING],
                drop_rate=params[param_keys.DROP_RATE],
                training=training,
                name='multi_stage_blstm')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            batchnorm=params[param_keys.BATCHNORM_FC],
            dropout=params[param_keys.DROPOUT_FC],
            drop_rate=params[param_keys.DROP_RATE],
            training=training,
            name='fc'
        )
        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities
