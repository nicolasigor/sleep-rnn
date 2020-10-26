from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_RESOURCES = os.path.join(PATH_THIS_DIR, '..', '..', 'resources')

import numpy as np
import tensorflow as tf

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