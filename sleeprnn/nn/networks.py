"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from . import layers
from . import cwt

from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys


class DummyNet(object):
    def __init__(self, params, training, name="model_dummy"):
        print('Using model Dummy')
        self.p = params
        self.training = training
        self.name = name
        # Useful constants
        self.crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

    def __call__(self, inputs):
        """Input [batch_size, time_len, n_channels]"""
        with tf.variable_scope(self.name):
            # Prepare Input
            inputs = inputs[:, self.crop:-self.crop, :]

            # Apply some layers
            outputs = tf.keras.layers.AvgPool1D(8)(inputs)

            logits = tf.keras.layers.Conv1D(
                filters=1, kernel_size=1, name="conv_logits",
                kernel_initializer=tf.initializers.he_normal()
            )(outputs)

            probas = tf.nn.sigmoid(logits, name="probabilities")

            # Define outputs
            output_dict = {
                constants.LOGITS: logits,
                constants.PROBABILITIES: probas
            }
        return output_dict


class RedTimeNet(object):
    def __init__(self, params, training, name="model_red_time"):
        print('Using model RED-Time')
        self.p = params
        self.training = training
        self.name = name
        # Useful constants
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        self.start_crop = border_crop
        self.end_crop = (-border_crop) if (border_crop > 0) else None

    def __call__(self, inputs):
        """Input [batch_size, time_len, n_channels]"""
        with tf.variable_scope(self.name):
            # Prepare Input
            inputs = inputs[:, self.start_crop:-self.end_crop, :]
            outputs = tf.keras.layers.BatchNormalization(
                name="bn_input")(inputs, training=self.training)

            # Local feature extraction
            for i in range(3):
                filters = self.p[pkeys.TIME_CONV_INITIAL_FILTERS] * (2 ** i)

                outputs = tf.keras.layers.Conv1D(
                    filters=filters, kernel_size=3, use_bias=False,
                    padding="same", name="conv_%d_a" % (i + 1),
                    kernel_initializer=tf.initializers.he_normal()
                )(outputs)

                outputs = tf.keras.layers.BatchNormalization(
                    name="bn_%d_a" % (i + 1), scale=False
                )(outputs, training=self.training)

                outputs = tf.nn.relu(outputs)

                outputs = tf.keras.layers.Conv1D(
                    filters=filters, kernel_size=3, use_bias=False,
                    padding="same", name="conv_%d_b" % (i + 1),
                    kernel_initializer=tf.initializers.he_normal()
                )(outputs)

                outputs = tf.keras.layers.BatchNormalization(
                    name="bn_%d_b" % (i + 1), scale=False
                )(outputs, training=self.training)

                outputs = tf.nn.relu(outputs)

                if self.p[pkeys.CONV_DOWNSAMPLING] == constants.AVGPOOL:
                    outputs = tf.keras.layers.AvgPool1D(2)(outputs)
                elif self.p[pkeys.CONV_DOWNSAMPLING == constants.MAXPOOL]:
                    outputs = tf.keras.layers.MaxPool1D(2)(outputs)
                else:
                    raise ValueError()

            # BLSTM Processing
            if self.p[pkeys.DROP_RATE_BEFORE_LSTM] > 0:
                outputs = tf.keras.layers.SpatialDropout1D(
                    self.p[pkeys.DROP_RATE_BEFORE_LSTM]
                )(outputs, training=self.training)

            outputs = layers.blstm_wrapper(
                outputs, num_units=self.p[pkeys.INITIAL_LSTM_UNITS],
                name="blstm_1")

            if self.p[pkeys.DROP_RATE_HIDDEN] > 0:
                outputs = tf.keras.layers.SpatialDropout1D(
                    self.p[pkeys.DROP_RATE_HIDDEN]
                )(outputs, training=self.training)

            outputs = layers.blstm_wrapper(
                outputs, num_units=self.p[pkeys.INITIAL_LSTM_UNITS],
                name="blstm_2")

            # Classification
            if self.p[pkeys.DROP_RATE_HIDDEN] > 0:
                outputs = tf.keras.layers.SpatialDropout1D(
                    self.p[pkeys.DROP_RATE_HIDDEN]
                )(outputs, training=self.training)

            outputs = tf.keras.layers.Conv1D(
                filters=self.p[pkeys.FC_UNITS], activation=tf.nn.relu,
                kernel_size=1, name="conv_fc",
                kernel_initializer=tf.initializers.he_normal()
            )(outputs)

            if self.p[pkeys.DROP_RATE_OUTPUT] > 0:
                outputs = tf.keras.layers.SpatialDropout1D(
                    self.p[pkeys.DROP_RATE_OUTPUT]
                )(outputs, training=self.training)

            logits = tf.keras.layers.Conv1D(
                filters=1, kernel_size=1, name="conv_logits",
                kernel_initializer=tf.initializers.he_normal()
            )(outputs)

            probas = tf.nn.sigmoid(logits, name="probabilities")

            # Define outputs
            output_dict = {
                constants.LOGITS: logits,
                constants.PROBABILITIES: probas
            }
        return output_dict


# TODO: re-implement Red-cwt, AttTime, test graph, and test training

class RedCwtNet(object):
    def __init__(self, params, training, name="model_red_cwt"):
        print('Using model RED-CWT')
        self.p = params
        self.training = training
        self.name = name
        # Useful constants
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])

    def __call__(self, inputs):
        """Input [batch_size, time_len, n_channels]"""
        with tf.variable_scope(self.name):
            pass


class AttTimeNet(object):
    def __init__(self, params, training, name="model_att_time"):
        print('Using model ATT-Time')
        self.p = params
        self.training = training
        self.name = name
        # Useful constants
        border_crop = int(
            params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        self.start_crop = border_crop
        self.end_crop = (-border_crop) if (border_crop > 0) else None

    def __call__(self, inputs):
        """Input [batch_size, time_len, n_channels]"""
        with tf.variable_scope(self.name):
            pass



def wavelet_blstm_net_att04(
        inputs,
        params,
        training,
        name='model_att04'
):
    print('Using model ATT04 (Time-Domain)')
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

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]

        with tf.variable_scope('attention'):

            # Multilayer BLSTM (2 layers)
            after_lstm_outputs = layers.multilayer_lstm_block(
                outputs,
                params[pkeys.ATT_LSTM_DIM],
                n_layers=2,
                num_dirs=constants.BIDIRECTIONAL,
                dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
                dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
                drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
                drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name='multi_layer_blstm')

            # Prepare input for values
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=att_dim,
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name='pos_enc'
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis

            v_outputs = layers.sequence_fc_layer(
                outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name='fc_embed_v')
            v_outputs = v_outputs + pos_enc
            v_outputs = layers.dropout_layer(
                v_outputs, 'drop_embed_v',
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training)

            # Prepare input for queries and keys
            qk_outputs = layers.sequence_fc_layer(
                after_lstm_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name='fc_embed_qk')
            qk_outputs = qk_outputs + pos_enc
            qk_outputs = layers.dropout_layer(
                qk_outputs, 'drop_embed_qk',
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training)

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name='queries')
            keys = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name='keys')
            values = layers.sequence_fc_layer(
                v_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name='values')

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values,
                n_heads,
                name='multi_head_att')

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                4 * params[pkeys.ATT_DIM],
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



