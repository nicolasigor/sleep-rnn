"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers

from utils import constants


def cmorlet_conv_blstm_net(
        inputs,
        fb_list,
        fs,
        border_crop,
        training,
        n_conv_blocks=0,
        n_time_levels=1,
        batchnorm_conv=constants.BN_RENORM,
        batchnorm_first_lstm=constants.BN_RENORM,
        dropout_first_lstm=None,
        batchnorm_rest_lstm=None,
        dropout_rest_lstm=None,
        time_pooling=constants.AVGPOOL,
        batchnorm_fc=None,
        dropout_fc=None,
        drop_rate=0.5,
        trainable_wavelet=False,
        name='model'):
    """ Basic model with cmorlet, convolutions, and 2-layers BLSTM.

    This models first computes the CWT using the cmorlet wavelets, then, if
    applicable, applies convolutional blocks of two 3x3 convolutions followed
    by maxpooling. After this, the outputs is flatten and is passed to a
    blstm stage. This stage is a 2-layers BLSTM is only one time level is used,
    or it's a ladder network downsampling and upsampling the time dimension if
    two or three time levels are used. The final classification is made with a
    FC layer with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        training: (boolean) Indicates if it is the training phase or not.
        n_conv_blocks: (Optional, {0, 1, 2, 3}, defaults to 0) Indicates the
            number of convolutional blocks to be performed after the CWT and
            before the BLSTM. If 0, no blocks are applied.
        n_time_levels: (Optional, {1, 2, 3}, defaults 1) Indicates the number
            of stages for the recurrent part, building a ladder-like network.
            If 1, it's a simple 2-layers BLSTM network.
        batchnorm_conv: (Optional, {None, BN, BN_RENORM}, defaults to BN_RENORM)
            Type of batchnorm to be used in the convolutional blocks. BN is
            normal batchnorm, and BN_RENORM is a batchnorm with renorm
            activated. If None, batchnorm is not applied.
        batchnorm_first_lstm: (Optional, {None, BN, BN_RENORM}, defaults to
            BN_RENORM) Type of batchnorm to be used in the first BLSTM layer.
            BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm
            activated. If None, batchnorm is not applied.
        dropout_first_lstm: (Optional, {None REGULAR_DROP, SEQUENCE_DROP},
            defaults to None) Type of dropout to be used in the first BLSTM
            layer. REGULAR_DROP is regular  dropout, and SEQUENCE_DROP is a
            dropout with the same noise shape for each time_step. If None,
            dropout is not applied. The dropout layer is applied after the
            batchnorm.
        batchnorm_rest_lstm: (Optional, {None, BN, BN_RENORM}, defaults to
            None) Type of batchnorm to be used in the rest of BLSTM layers.
            BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm
            activated. If None, batchnorm is not applied.
        dropout_rest_lstm: (Optional, {None REGULAR_DROP, SEQUENCE_DROP},
            defaults to None) Type of dropout to be used in the rest of BLSTM
            layers. REGULAR_DROP is regular  dropout, and SEQUENCE_DROP is a
            dropout with the same noise shape for each time_step. If None,
            dropout is not applied. The dropout layer is applied after the
            batchnorm.
        time_pooling: (Optional, {AVGPOOL, MAXPOOL}, defaults to AVGPOOL)
            Indicates the type of pooling to be performed to downsample
            the time dimension if n_time_levels > 1.
        batchnorm_fc: (Optional, {None, BN, BN_RENORM}, defaults to
            None) Type of batchnorm to be used in the output layer.
            BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm
            activated. If None, batchnorm is not applied.
        dropout_fc: (Optional, {None REGULAR_DROP, SEQUENCE_DROP},
            defaults to None) Type of dropout to be used in output
            layer. REGULAR_DROP is regular  dropout, and SEQUENCE_DROP is a
            dropout with the same noise shape for each time_step. If None,
            dropout is not applied. The dropout layer is applied after the
            batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        trainable_wavelet: (Optional, boolean, defaults to False) If True, the
            fb params will be trained with backprop.
        name: (Optional, string, defaults to 'model') A name for the network.
    """
    with tf.variable_scope(name):

        if n_conv_blocks not in [0, 1, 2, 3]:
            msg = constants.ERROR_INVALID \
                  % ([0, 1, 2, 3], 'n_conv_blocks', n_conv_blocks)
            raise ValueError(msg)
        if n_time_levels not in [1, 2, 3]:
            msg = constants.ERROR_INVALID \
                  % ([1, 2, 3], 'n_time_levels', n_time_levels)
            raise ValueError(msg)

        cwt_stride = 8 / (2**n_conv_blocks)

        # CWT CMORLET
        outputs = layers.cmorlet_layer(
            inputs,
            fb_list,
            fs,
            lower_freq=1,
            upper_freq=30,
            n_scales=32,
            stride=cwt_stride,
            border_crop=border_crop,
            training=training,
            trainable_wavelet=trainable_wavelet,
            name='spectrum')

        # Convolutional stage (only if n_conv_blocks is greater than 0)
        for i in range(n_conv_blocks):
            filters = 16 * (2**i)
            outputs = layers.bn_conv3_block(
                outputs,
                filters,
                batchnorm=batchnorm_conv,
                training=training,
                name='conv_block_%d' % (i+1))

        outputs = layers.sequence_flatten(outputs, 'flatten')

        # Multi stage BLSTM
        if n_time_levels == 1:
            first_level_channels = 256

            # Just a simple 2-layers BLSTM
            outputs = layers.lstm_layer(
                outputs,
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_first_lstm,
                dropout=dropout_first_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_1')
            outputs = layers.lstm_layer(
                outputs,
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_2')

        elif n_time_levels == 2:
            first_level_channels = 128
            second_level_channels = 256

            # Go down
            outputs_1e = layers.lstm_layer(
                outputs,
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_first_lstm,
                dropout=dropout_first_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_1e')
            outputs_1e_down = layers.time_downsampling_layer(
                outputs_1e, pooling=time_pooling, name='down_1e')

            outputs_deep = layers.lstm_layer(
                outputs_1e_down,
                second_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_deep')

            # Go up
            outputs_1d_up = layers.time_upsampling_layer(
                outputs_deep, first_level_channels, name='up_1e')
            outputs = layers.lstm_layer(
                tf.concat([outputs_1d_up, outputs_1e], axis=-1),
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_1d')

        else:  # it's 3, we need to go deeper
            first_level_channels = 128
            second_level_channels = 256
            third_level_channels = 512

            # Go down
            outputs_1e = layers.lstm_layer(
                outputs,
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_first_lstm,
                dropout=dropout_first_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_1e')
            outputs_1e_down = layers.time_downsampling_layer(
                outputs_1e, pooling=time_pooling, name='down_1e')
            outputs_2e = layers.lstm_layer(
                outputs_1e_down,
                second_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_2e')
            outputs_2e_down = layers.time_downsampling_layer(
                outputs_2e, pooling=time_pooling, name='down_2e')

            outputs_deep = layers.lstm_layer(
                outputs_2e_down,
                third_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_deep')

            # Go up
            outputs_2d_up = layers.time_upsampling_layer(
                outputs_deep, second_level_channels, name='up_2d')
            outputs_2d = layers.lstm_layer(
                tf.concat([outputs_2d_up, outputs_2e], axis=-1),
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_2d')
            outputs_1d_up = layers.time_upsampling_layer(
                outputs_2d, first_level_channels, name='up_1d')
            outputs = layers.lstm_layer(
                tf.concat([outputs_1d_up, outputs_1e], axis=-1),
                first_level_channels // 2,
                num_dirs=constants.BIDIRECTIONAL,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='blstm_1d')

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            batchnorm=batchnorm_fc,
            dropout=dropout_fc,
            drop_rate=drop_rate,
            training=training,
            name='fc'
        )
        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities
