from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_RESOURCES = os.path.join(PATH_THIS_DIR, '..', '..', 'resources')

import numpy as np
import tensorflow as tf

from .expert_feats import a7_layer_tf, bandpass_tf_batch
from . import layers
from . import spectrum

from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys


def wavelet_blstm_net_att05(
        inputs,
        params,
        training,
        name='model_att05'
):
    print('Using model ATT05 (CWT-Domain)')
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

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
        outputs_flatten = layers.sequence_flatten(outputs, 'flatten')

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]

        # bands parameters
        v_add_band_enc = params[pkeys.ATT_BANDS_V_ADD_BAND_ENC]
        k_add_band_enc = params[pkeys.ATT_BANDS_K_ADD_BAND_ENC]
        v_indep_linear = params[pkeys.ATT_BANDS_V_INDEP_LINEAR]
        k_indep_linear = params[pkeys.ATT_BANDS_K_INDEP_LINEAR]
        n_bands = params[pkeys.N_SCALES] // 4

        with tf.variable_scope('attention'):

            # Multilayer BLSTM (2 layers)
            after_lstm_outputs = layers.multilayer_lstm_block(
                outputs_flatten,
                params[pkeys.ATT_LSTM_DIM],
                n_layers=2,
                num_dirs=constants.BIDIRECTIONAL,
                dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
                dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
                drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
                drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name='multi_layer_blstm')

            # Prepare positional encoding
            with tf.variable_scope("pos_enc"):
                pos_enc = layers.get_positional_encoding(
                    seq_len=seq_len,
                    dims=att_dim,
                    pe_factor=params[pkeys.ATT_PE_FACTOR],
                    name='pos_enc')
                pos_enc_1d = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
                pos_enc = tf.expand_dims(pos_enc_1d, axis=2)  # Add band axis
                # shape [1, time, 1, dim]

            # Prepare band encoding
            if v_add_band_enc or k_add_band_enc:
                with tf.variable_scope("band_enc"):
                    bands_labels = list(range(n_bands))
                    bands_oh = tf.one_hot(bands_labels, depth=n_bands)
                    bands_oh = tf.expand_dims(bands_oh, axis=0)  # Add batch axis
                    # shape [1, n_bands, n_bands]
                    bands_enc = layers.sequence_fc_layer(
                        bands_oh,
                        att_dim,
                        kernel_init=tf.initializers.he_normal(),
                        training=training, use_bias=False,
                        name='band_enc')
                    # shape [1, n_bands, dim]
                    bands_enc = tf.expand_dims(bands_enc, axis=1)  # Add time axis
                    # shape [1, 1, n_bands, dim]

            # Prepare input for values
            with tf.variable_scope("values_prep"):
                # input shape [batch, time, n_bands, feats]
                if v_indep_linear:  # indep projections
                    with tf.variable_scope("fc_embed_v_indep"):
                        outputs_unstack = tf.unstack(outputs, axis=2)
                        projected_list = []
                        for i, output_band in enumerate(outputs_unstack):
                            output_band = tf.expand_dims(output_band, axis=2)
                            # shape [batch, time, 1, feats]
                            output_band = tf.layers.conv2d(
                                inputs=output_band, filters=att_dim, kernel_size=1,
                                name='fc_embed_v_%d' % i, use_bias=False,
                                kernel_initializer=tf.initializers.he_normal())
                            projected_list.append(output_band)
                            # shape [batch, time, 1, dim]
                        v_outputs = tf.concat(projected_list, axis=2)
                else:  # shared projection
                    v_outputs = tf.layers.conv2d(
                        inputs=outputs, filters=att_dim, kernel_size=1,
                        name='fc_embed_v', use_bias=False,
                        kernel_initializer=tf.initializers.he_normal())
                v_outputs = v_outputs + pos_enc
                if v_add_band_enc:
                    v_outputs = v_outputs + bands_enc
                # shape [batch, time, n_bands, dim]
                v_outputs = tf.reshape(
                    v_outputs,
                    shape=(-1, seq_len * n_bands, att_dim),
                    name="flatten_bands_v")
                # shape [batch, time * n_bands, dim]
                v_outputs = layers.dropout_layer(
                    v_outputs, 'drop_embed_v',
                    drop_rate=params[pkeys.ATT_DROP_RATE],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training)
                # shape [batch, time * n_bands, dim]

            # Prepare input for queries
            with tf.variable_scope("queries_prep"):
                q_outputs = layers.sequence_fc_layer(
                    after_lstm_outputs,
                    att_dim,
                    kernel_init=tf.initializers.he_normal(),
                    training=training, use_bias=False,
                    name='fc_embed_q')
                q_outputs = q_outputs + pos_enc_1d
                q_outputs = layers.dropout_layer(
                    q_outputs, 'drop_embed_q',
                    drop_rate=params[pkeys.ATT_DROP_RATE],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training)
                # shape [batch, time, dim]

            # Prepare input for keys
            with tf.variable_scope("keys_prep"):
                after_lstm_outputs_2d = tf.expand_dims(after_lstm_outputs, axis=2)
                # input shape [batch, time, 1, dim]
                if k_indep_linear:  # indep projections
                    with tf.variable_scope("fc_embed_k_indep"):
                        projected_list = []
                        for i in range(n_bands):
                            output_band = tf.layers.conv2d(
                                inputs=after_lstm_outputs_2d, filters=att_dim, kernel_size=1,
                                name='fc_embed_k_%d' % i, use_bias=False,
                                kernel_initializer=tf.initializers.he_normal())
                            projected_list.append(output_band)
                            # shape [batch, time, 1, dim]
                        k_outputs = tf.concat(projected_list, axis=2)
                        # shape [batch, time, n_bands, dim]
                else:  # shared projection
                    k_outputs = tf.layers.conv2d(
                        inputs=after_lstm_outputs_2d, filters=att_dim, kernel_size=1,
                        name='fc_embed_k', use_bias=False,
                        kernel_initializer=tf.initializers.he_normal())
                    # shape [batch, time, 1, dim]
                k_outputs = k_outputs + pos_enc
                if k_add_band_enc:
                    k_outputs = k_outputs + bands_enc
                # shape [batch, time, n_bands, dim]
                k_outputs = tf.reshape(
                    k_outputs, shape=(-1, seq_len * n_bands, att_dim),
                    name="flatten_bands_k")
                # shape [batch, time * n_bands, dim]
                k_outputs = layers.dropout_layer(
                    k_outputs, 'drop_embed_k',
                    drop_rate=params[pkeys.ATT_DROP_RATE],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training)
                # shape [batch, time * n_bands, dim]

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                q_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training, use_bias=False,
                name='queries')
            keys = layers.sequence_fc_layer(
                k_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training, use_bias=False,
                name='keys')
            values = layers.sequence_fc_layer(
                v_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training, use_bias=False,
                name='values')

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values,
                n_heads,
                name='multi_head_att')
            # should be [batch, time, dim]

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                2 * params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ATT_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name='ffn_1')

            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name='ffn_2')

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def deep_a7_v1(
        inputs,
        params,
        training,
        name='model_a7_v1'
):
    print('Using model A7_V1 (A7 feats, convolutional)')
    with tf.variable_scope(name):
        # input is [batch, time_len]
        inputs = a7_layer_tf(
            inputs,
            fs=params[pkeys.FS],
            window_duration=params[pkeys.A7_WINDOW_DURATION],
            window_duration_relSigPow=params[pkeys.A7_WINDOW_DURATION_REL_SIG_POW],
            use_log_absSigPow=params[pkeys.A7_USE_LOG_ABS_SIG_POW],
            use_log_relSigPow=params[pkeys.A7_USE_LOG_REL_SIG_POW],
            use_log_sigCov=params[pkeys.A7_USE_LOG_SIG_COV],
            use_log_sigCorr=params[pkeys.A7_USE_LOG_SIG_CORR],
            use_zscore_relSigPow=params[pkeys.A7_USE_ZSCORE_REL_SIG_POW],
            use_zscore_sigCov=params[pkeys.A7_USE_ZSCORE_SIG_COV],
            use_zscore_sigCorr=params[pkeys.A7_USE_ZSCORE_SIG_CORR],
            remove_delta_in_cov=params[pkeys.A7_REMOVE_DELTA_IN_COV],
            dispersion_mode=params[pkeys.A7_DISPERSION_MODE]
        )

        # Now is [batch, time_len, 4]
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop, :]

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # Now pool to get proper length
        outputs = tf.keras.layers.AveragePooling1D(pool_size=8)(outputs)

        # Now convolutions
        kernel_size = params[pkeys.A7_CNN_KERNEL_SIZE]
        n_layers = params[pkeys.A7_CNN_N_LAYERS]
        filters = params[pkeys.A7_CNN_FILTERS]
        drop_rate_conv = params[pkeys.A7_CNN_DROP_RATE]
        for i in range(n_layers):
            with tf.variable_scope('conv_%d' % i):
                if i > 0 and drop_rate_conv > 0:
                    outputs = layers.dropout_layer(
                        outputs, 'drop_%d' % i, training, dropout=params[pkeys.TYPE_DROPOUT], drop_rate=drop_rate_conv)
                outputs = tf.keras.layers.Conv1D(
                    filters=filters, kernel_size=kernel_size, padding='same', use_bias=False,
                    kernel_initializer=tf.initializers.he_normal())(outputs)
                outputs = layers.batchnorm_layer(
                    outputs, 'bn_%d' % i, batchnorm=params[pkeys.TYPE_BATCHNORM], training=training, scale=False)
                outputs = tf.nn.relu(outputs)

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def deep_a7_v2(
        inputs,
        params,
        training,
        name='model_a7_v2'
):
    print('Using model A7_V2 (A7 feats, recurrent)')
    with tf.variable_scope(name):
        # input is [batch, time_len]
        inputs = a7_layer_tf(
            inputs,
            fs=params[pkeys.FS],
            window_duration=params[pkeys.A7_WINDOW_DURATION],
            window_duration_relSigPow=params[pkeys.A7_WINDOW_DURATION_REL_SIG_POW],
            use_log_absSigPow=params[pkeys.A7_USE_LOG_ABS_SIG_POW],
            use_log_relSigPow=params[pkeys.A7_USE_LOG_REL_SIG_POW],
            use_log_sigCov=params[pkeys.A7_USE_LOG_SIG_COV],
            use_log_sigCorr=params[pkeys.A7_USE_LOG_SIG_CORR],
            use_zscore_relSigPow=params[pkeys.A7_USE_ZSCORE_REL_SIG_POW],
            use_zscore_sigCov=params[pkeys.A7_USE_ZSCORE_SIG_COV],
            use_zscore_sigCorr=params[pkeys.A7_USE_ZSCORE_SIG_CORR],
            remove_delta_in_cov=params[pkeys.A7_REMOVE_DELTA_IN_COV],
            dispersion_mode=params[pkeys.A7_DISPERSION_MODE]
        )

        # Now is [batch, time_len, 4]
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop, :]

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # Now pool to get proper length
        outputs = tf.keras.layers.AveragePooling1D(pool_size=8)(outputs)

        # Now recurrent
        lstm_units = params[pkeys.A7_RNN_LSTM_UNITS]
        fc_units = params[pkeys.A7_RNN_FC_UNITS]
        drop_rate_hidden = params[pkeys.A7_RNN_DROP_RATE]

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            lstm_units,
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=0,
            drop_rate_rest_lstm=drop_rate_hidden,
            training=training,
            name='multi_layer_blstm')

        if fc_units > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                fc_units,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=drop_rate_hidden,
                training=training,
                activation=tf.nn.relu,
                name='fc_1')

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def deep_a7_v3(
        inputs,
        params,
        training,
        name='model_a7_v3'
):
    print('Using model A7_V3 (A7 feats input, RED architecture)')
    with tf.variable_scope(name):
        # input is [batch, time_len]
        inputs = a7_layer_tf(
            inputs,
            fs=params[pkeys.FS],
            window_duration=params[pkeys.A7_WINDOW_DURATION],
            window_duration_relSigPow=params[pkeys.A7_WINDOW_DURATION_REL_SIG_POW],
            use_log_absSigPow=params[pkeys.A7_USE_LOG_ABS_SIG_POW],
            use_log_relSigPow=params[pkeys.A7_USE_LOG_REL_SIG_POW],
            use_log_sigCov=params[pkeys.A7_USE_LOG_SIG_COV],
            use_log_sigCorr=params[pkeys.A7_USE_LOG_SIG_CORR],
            use_zscore_relSigPow=params[pkeys.A7_USE_ZSCORE_REL_SIG_POW],
            use_zscore_sigCov=params[pkeys.A7_USE_ZSCORE_SIG_COV],
            use_zscore_sigCorr=params[pkeys.A7_USE_ZSCORE_SIG_CORR],
            remove_delta_in_cov=params[pkeys.A7_REMOVE_DELTA_IN_COV],
            dispersion_mode=params[pkeys.A7_DISPERSION_MODE]
        )

        # Now is [batch, time_len, 4]
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop, :]

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_bp(
        inputs,
        params,
        training,
        name='model_v11_bp'
):
    print('Using model V11-BP (Time-Domain on band-passed signal)')
    with tf.variable_scope(name):
        # Band-pass
        print("Applying bandpass between %s Hz and %s Hz" % (
            params[pkeys.BP_INPUT_LOWCUT], params[pkeys.BP_INPUT_HIGHCUT]))
        inputs = bandpass_tf_batch(
            inputs,
            fs=params[pkeys.FS],
            lowcut=params[pkeys.BP_INPUT_LOWCUT],
            highcut=params[pkeys.BP_INPUT_HIGHCUT])

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_bp(
        inputs,
        params,
        training,
        name='model_v19_bp'
):
    print('Using model V19-BP (general cwt on band-passed signal)')
    with tf.variable_scope(name):
        # Band-pass
        print("Applying bandpass between %s Hz and %s Hz" % (
            params[pkeys.BP_INPUT_LOWCUT], params[pkeys.BP_INPUT_HIGHCUT]))
        inputs = bandpass_tf_batch(
            inputs,
            fs=params[pkeys.FS],
            lowcut=params[pkeys.BP_INPUT_LOWCUT],
            highcut=params[pkeys.BP_INPUT_HIGHCUT])

        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_ln(
        inputs,
        params,
        training,
        name='model_v11_ln'
):
    print('Using model V11-LN (Time-Domain with zscore at conv)')
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

        outputs = layers.conv1d_prebn_block_with_zscore(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_3_zscore')

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_ln2(
        inputs,
        params,
        training,
        name='model_v11_ln2'
):
    print('Using model V11-LN2 (Time-Domain with zscore at logits)')
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
            outputs = layers.sequence_fc_layer_with_zscore(
                outputs,
                params[pkeys.FC_UNITS],
                training=training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                kernel_init=tf.initializers.he_normal(),
                name='fc_1_zscore')

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_ln2(
        inputs,
        params,
        training,
        name='model_v19_ln2'
):
    print('Using model V19-LN2 (general cwt with zscore at logits)')
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name='spectrum')

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
            outputs = layers.sequence_fc_layer_with_zscore(
                outputs,
                params[pkeys.FC_UNITS],
                training=training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                kernel_init=tf.initializers.he_normal(),
                name='fc_1_zscore')

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_ln3(
        inputs,
        params,
        training,
        name='model_v11_ln3'
):
    print('Using model V11-LN3 (Time-Domain with zscore at logits and last conv)')
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

        outputs = layers.conv1d_prebn_block_with_zscore(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name='convblock_1d_3_zscore')

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
            outputs = layers.sequence_fc_layer_with_zscore(
                outputs,
                params[pkeys.FC_UNITS],
                training=training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                kernel_init=tf.initializers.he_normal(),
                name='fc_1_zscore')

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_mk(
        inputs,
        params,
        training,
        name='model_v11_mk'
):
    print('Using model V11-MK (Time-Domain with multi-kernel convolutions)')
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters in params[pkeys.TIME_CONV_MK_FILTERS_1]:
            print("    k %d and f %d" % (kernel_size, n_filters))
            tmp_out = layers.conv1d_prebn_block_with_projection(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                project_first=params[pkeys.TIME_CONV_MK_PROJECT_FIRST],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_1' % kernel_size)
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters in params[pkeys.TIME_CONV_MK_FILTERS_2]:
            print("    k %d and f %d" % (kernel_size, n_filters))
            tmp_out = layers.conv1d_prebn_block_with_projection(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                project_first=params[pkeys.TIME_CONV_MK_PROJECT_FIRST],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_2' % kernel_size)
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters in params[pkeys.TIME_CONV_MK_FILTERS_3]:
            print("    k %d and f %d" % (kernel_size, n_filters))
            tmp_out = layers.conv1d_prebn_block_with_projection(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                project_first=params[pkeys.TIME_CONV_MK_PROJECT_FIRST],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_3' % kernel_size)
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(inputs=outputs_1, pool_size=(4, 1), strides=(4, 1))
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(inputs=outputs_2, pool_size=(2, 1), strides=(2, 1))
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_mkd(
        inputs,
        params,
        training,
        name='model_v11_mkd'
):
    print('Using model V11-MKD (Time-Domain with multi-dilated convolutions)')
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_1' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_2' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_3' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(inputs=outputs_1, pool_size=(4, 1), strides=(4, 1))
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(inputs=outputs_2, pool_size=(2, 1), strides=(2, 1))
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_mkd2(
        inputs,
        params,
        training,
        name='model_v11_mkd2'
):
    print('Using model V11-MKD-2 (Time-Domain with multi-dilated convolutions, border crop AFTER lstm)')
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_1' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_2' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_3' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(inputs=outputs_1, pool_size=(4, 1), strides=(4, 1))
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(inputs=outputs_2, pool_size=(2, 1), strides=(2, 1))
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_duration_to_crop_after_conv = 1
        border_duration_to_crop_after_lstm = params[pkeys.BORDER_DURATION] - border_duration_to_crop_after_conv

        border_crop = int(border_duration_to_crop_after_conv * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_duration_to_crop_after_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name='logits')

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def stat_net(
        inputs_normalized,
        params,
        training,
        name='stat_net_conv'
):
    print('Using DOSED-like statistics network')
    with tf.variable_scope(name):

        n_layers = params[pkeys.STAT_NET_DEPTH]
        kernel_size = params[pkeys.STAT_NET_KERNEL_SIZE]
        type_pool = params[pkeys.STAT_NET_TYPE_POOL]
        drop_rate = params[pkeys.STAT_NET_DROP_RATE]
        output_dim = params[pkeys.STAT_NET_OUTPUT_DIM]
        max_filters = params[pkeys.STAT_NET_MAX_FILTERS]

        batchnorm = params[pkeys.TYPE_BATCHNORM]

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        outputs = tf.expand_dims(inputs_normalized, axis=2)
        use_bias = batchnorm is None

        for i in range(1, n_layers+1):
            with tf.variable_scope('conv%d_%d' % (kernel_size, i)):
                filters = 2 ** (2 + i)
                filters = min(filters, max_filters)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
                    padding=constants.PAD_VALID, name='conv%d' % kernel_size,
                    kernel_initializer=tf.initializers.he_normal(), use_bias=use_bias)
                if batchnorm:
                    outputs = layers.batchnorm_layer(
                        outputs, 'bn', batchnorm=batchnorm, training=training, scale=False)
                outputs = tf.nn.relu(outputs)
                outputs = layers.pooling1d(outputs, type_pool)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
        outputs = tf.keras.layers.GlobalAvgPool1D()(outputs)
        outputs = tf.layers.dropout(outputs, training=training, rate=drop_rate)
        outputs = tf.keras.layers.Dense(output_dim, kernel_initializer=tf.initializers.he_normal())(outputs)
    return outputs


def stat_net_lstm(
        inputs_normalized,
        params,
        training,
        name='stat_net_lstm'
):
    print('Using BLSTM statistics network')
    with tf.variable_scope(name):
        output_dim = params[pkeys.STAT_NET_OUTPUT_DIM]
        drop_rate = params[pkeys.STAT_NET_DROP_RATE]

        outputs = layers.lstm_layer(
            inputs_normalized,
            num_units=params[pkeys.STAT_NET_LSTM_UNITS],
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name='blstm')
        outputs = tf.keras.layers.GlobalAvgPool1D()(outputs)
        outputs = tf.layers.dropout(outputs, training=training, rate=drop_rate)
        outputs = tf.keras.layers.Dense(output_dim, kernel_initializer=tf.initializers.he_normal())(outputs)
    return outputs


def modulation_net(
        inputs_normalized,
        output_size,
        params,
        training,
        use_scale=True,
        use_bias=True,
        name='stat_net_mod'
):
    learnable_scale = 1.0
    learnable_bias = 0.0
    if use_scale or use_bias:
        print('Using modulation networks')
        with tf.variable_scope(name):
            backbone = params[pkeys.STAT_MOD_NET_TYPE_BACKBONE]
            if backbone == 'conv':
                outputs = stat_net(inputs_normalized, params, training)
            elif backbone == 'lstm':
                outputs = stat_net_lstm(inputs_normalized, params, training)
            else:
                raise ValueError('%s not a valid backbone type' % backbone)
            # [batch, n_feats] -> [batch, 1, n_feats]
            outputs = tf.expand_dims(outputs, axis=1)
            outputs = tf.nn.relu(outputs)
            # Scale and bias
            if use_scale:
                learnable_scale += layers.sequence_fc_layer(
                    outputs,
                    output_size,
                    kernel_init=tf.initializers.he_normal(),
                    training=training,
                    use_bias=False,
                    name='learn_scale')
            if use_bias:
                learnable_bias += layers.sequence_fc_layer(
                    outputs,
                    output_size,
                    kernel_init=tf.initializers.he_normal(),
                    training=training,
                    use_bias=False,
                    name='learn_bias')
    else:
        print("Not using modulation")
    return learnable_scale, learnable_bias


def segment_net(
        inputs_normalized,
        params,
        training,
        output_activation=tf.nn.relu,
        border_conv=1,
        border_lstm=5,
        name='segment_net'
):
    print("Using V11-MKD2 segment network")
    with tf.variable_scope(name):
        # 1D convolutions expect shape [batch, time_len, n_feats]
        outputs = inputs_normalized

        # Only keep border_conv + border_lstm
        border_duration_to_keep = border_conv + border_lstm
        border_duration_to_crop = params[pkeys.BORDER_DURATION] - border_duration_to_keep
        border_crop = int(border_duration_to_crop * params[pkeys.FS])
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Convolutional stage (standard feed-forward)
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_1' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_2' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name='convblock_1d_k%d_d%d_3' % (kernel_size, dilation))
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(inputs=outputs_1, pool_size=(4, 1), strides=(4, 1))
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(inputs=outputs_2, pool_size=(2, 1), strides=(2, 1))
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        # Only keep border_lstm
        border_duration_to_crop = border_duration_to_keep - border_lstm
        border_crop = int(border_duration_to_crop * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        outputs = layers.sequence_fc_layer(
            outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=output_activation,
            name='fc')
    return outputs


def wavelet_blstm_net_v11_mkd2_stat(
        inputs,
        params,
        training,
        name='model_v11_mkd2_stat'
):
    print('Using model V11-MKD-2-STAT'
          '(Time-Domain with multi-dilated convolutions, border crop AFTER lstm, stat net)')
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        inputs = layers.batchnorm_layer(
            inputs, 'bn_input',
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training)
        # stat network
        outputs_context = stat_net(inputs, params, training)
        # [batch, n_feats] -> [batch, 1, n_feats]
        outputs_context = tf.expand_dims(outputs_context, axis=1)
        outputs_context = tf.nn.relu(outputs_context)
        # segment network
        outputs_segment = segment_net(inputs, params, training)

        if params[pkeys.STAT_NET_AFTER_CONCAT_FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs_segment = layers.sequence_fc_layer(
                outputs_segment,
                params[pkeys.STAT_NET_AFTER_CONCAT_FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name='fc_segment')
            outputs_context = layers.sequence_fc_layer(
                outputs_context,
                params[pkeys.STAT_NET_AFTER_CONCAT_FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                use_bias=False,
                name='fc_context')
            outputs = tf.nn.relu(outputs_segment + outputs_context)
            # Final FC classification layer
            logits = layers.sequence_output_2class_layer(
                outputs,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_OUTPUT],
                training=training,
                init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
                name='logits')
        else:
            # Final FC classification layer
            logits_segment = layers.sequence_output_2class_layer(
                outputs_segment,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_OUTPUT],
                training=training,
                init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
                name='logits_segment')
            logits_context = layers.sequence_fc_layer(
                outputs_context, 2,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_OUTPUT],
                training=training,
                use_bias=False,
                name='logits_context')
            logits = logits_segment + logits_context

        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_mkd2_statmod(
        inputs,
        params,
        training,
        name='model_v11_mkd2_statmod'
):
    print('Using model V11-MKD-2-STAT-MOD'
          '(Time-Domain with multi-dilated convolutions, border crop AFTER lstm, stat net modulation)')
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        inputs = layers.batchnorm_layer(inputs, 'bn_input', batchnorm=params[pkeys.TYPE_BATCHNORM], training=training)
        outputs = segment_net(inputs, params, training, output_activation=None)
        if params[pkeys.STAT_MOD_NET_MODULATE_LOGITS]:
            print("Modulating logits")
            outputs = tf.nn.relu(outputs)
            logits = layers.sequence_output_2class_layer(
                outputs,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_OUTPUT],
                training=training,
                init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
                name='logits_segment')
            # Modulate
            learnable_scale, learnable_bias = modulation_net(
                inputs, 2, params, training,
                use_scale=params[pkeys.STAT_MOD_NET_USE_SCALE], use_bias=params[pkeys.STAT_MOD_NET_USE_BIAS])
            logits = learnable_scale * logits + learnable_bias
        else:
            print("Modulating last hidden layer")
            learnable_scale, learnable_bias = modulation_net(
                inputs, params[pkeys.FC_UNITS], params, training,
                use_scale=params[pkeys.STAT_MOD_NET_USE_SCALE], use_bias=params[pkeys.STAT_MOD_NET_USE_BIAS])
            outputs = learnable_scale * outputs + learnable_bias
            outputs = tf.nn.relu(outputs)
            logits = layers.sequence_output_2class_layer(
                outputs,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_OUTPUT],
                training=training,
                init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
                name='logits_segment')
        with tf.variable_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram('probabilities', probabilities)
        cwt_prebn = None
    return logits, probabilities, cwt_prebn





