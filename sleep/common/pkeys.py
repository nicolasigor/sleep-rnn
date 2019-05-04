"""Module that stores several useful keys to configure a model.
"""

from . import constants

""" Input pipeline params

batch_size: (int) Size of the mini batches used for training.
shuffle_buffer_size: (int) Size of the buffer to shuffle the data. If 0, no 
    shuffle is applied.
prefetch_buffer_size: (int) Size of the buffer to prefetch the batches. If 0, 
    no prefetch is applied.
page_duration: (int) Size of a EEG page in seconds.
"""
BATCH_SIZE = 'batch_size'
SHUFFLE_BUFFER_SIZE = 'shuffle_buffer_size'
PREFETCH_BUFFER_SIZE = 'prefetch_buffer_size'
PAGE_DURATION = 'page_duration'

""" Model params
time_resolution_factor: (int) The original sampling frequency for the labels
    is downsampled using this factor.
fs: (float) Sampling frequency of the signals of interest.
border_duration: (int) Non-negative integer that
    specifies the number of seconds to be removed at each border at the
    end. This parameter allows to input a longer signal than the final
    desired size to remove border effects of the CWT.
fb_list: (list of floats) list of values for Fb (one for each scalogram)
n_conv_blocks: ({0, 1, 2, 3}) Indicates the
    number of convolutional blocks to be performed after the CWT and
    before the BLSTM. If 0, no blocks are applied.
n_time_levels: ({1, 2, 3}) Indicates the number
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
type_wavelet: ({CMORLET, SPLINE}) Type of wavelet to be used.
use_log: (boolean) Whether to apply logarithm to the CWT output, after the 
    avg pool.
n_scales: (int) Number of scales to be computed for the scalogram.
lower_freq: (float) Lower frequency to be considered in the scalograms [Hz].
upper_freq: (float) Upper frequency to be considered in the scalograms [Hz].
initial_conv_filters: (int) Number of filters to be used in the first
    convolutional block, if applicable. Subsequent conv blocks will double the
    previous number of filters.
initial_lstm_units: (int) Number of units for lstm layers. If multi stage
    is used (n_time_levels > 1), after every time downsampling operation 
    the number of units is doubled.
"""
# General parameters
FS = 'fs'
MODEL_VERSION = 'model_version'
BORDER_DURATION = 'border_duration'
# Regularization
TYPE_BATCHNORM = 'batchnorm'
TYPE_DROPOUT = 'dropout'
DROP_RATE_BEFORE_LSTM = 'drop_rate_before_lstm'
DROP_RATE_HIDDEN = 'drop_rate_hidden'
DROP_RATE_OUTPUT = 'drop_rate_output'
# CWT stage
FB_LIST = 'fb_list'
TRAINABLE_WAVELET = 'trainable_wavelet'
WAVELET_SIZE_FACTOR = 'wavelet_size_factor'
USE_LOG = 'use_log'
N_SCALES = 'n_scales'
LOWER_FREQ = 'lower_freq'
UPPER_FREQ = 'upper_freq'
USE_RELU = 'use_relu'
# Convolutional stage
INITIAL_KERNEL_SIZE = 'initial_kernel_size'
INITIAL_CONV_FILTERS = 'initial_conv_filters'
CONV_DOWNSAMPLING = 'conv_downsampling'
# blstm stage
INITIAL_LSTM_UNITS = 'initial_lstm_units'
# FC units in second to last layer
FC_UNITS = 'fc_units'


""" Loss params

class_weights: ({None, BALANCED, array_like}) Determines the class
    weights to be applied when computing the loss. If None, no weights
    are applied. If BALANCED, the weights balance the class
    frequencies. If is an array of shape [2,], class_weights[i]
    is the weight applied to class i.
type_loss: ({CROSS_ENTROPY_LOSS, DICE_LOSS}) Type of loss to be used 
"""
CLASS_WEIGHTS = 'class_weights'
TYPE_LOSS = 'type_loss'


""" Optimizer params

learning_rate: (float) learning rate for the optimizer
clip_norm: (float) this is the global norm to use to clip gradients. If None,
    no clipping is applied.
momentum: (float) momentum for the SGD optimizer.
use_nesterov: (bool) whether to use nesterov momentum instead of regular
    momentum for SGD optimization.
type_optimizer: ({ADAM_OPTIMIZER, SGD_OPTIMIZER}) Type of optimizer to be used.
"""
LEARNING_RATE = 'learning_rate'
CLIP_NORM = 'clip_norm'
MOMENTUM = 'momentum'
USE_NESTEROV_MOMENTUM = 'use_nesterov'
TYPE_OPTIMIZER = 'type_optimizer'


""" Training params

max_epochs: (int) Maximum numer of epochs to be performed in the training loop.
nstats: (int) Frequency in iterations to display metrics.
"""
MAX_ITERS = 'max_iters'
ITERS_STATS = 'iters_stats'
ITERS_LR_UPDATE = 'iters_lr_update'
REL_TOL_CRITERION = 'rel_tol_criterion'
LR_UPDATE_FACTOR = 'lr_update_factor'
LR_UPDATE_CRITERION = 'lr_update_criterion'
MAX_LR_UPDATES = 'max_lr_updates'


""" Postprocessing params 
"""
TOTAL_DOWNSAMPLING_FACTOR = 'total_downsampling_factor'
SS_MIN_SEPARATION = 'ss_min_separation'
SS_MIN_DURATION = 'ss_min_duration'
SS_MAX_DURATION = 'ss_max_duration'
KC_MIN_SEPARATION = 'kc_min_separation'
KC_MIN_DURATION = 'kc_min_duration'
KC_MAX_DURATION = 'kc_max_duration'

"""Inference params"""
PREDICT_WITH_AUGMENTED_PAGE = 'predict_with_augmented_page'


# Default parameters dictionary
default_params = {
    FS: 200,
    BATCH_SIZE: 32,
    SHUFFLE_BUFFER_SIZE: 1000,
    PREFETCH_BUFFER_SIZE: 2,
    PAGE_DURATION: 20,
    MODEL_VERSION: constants.V4,
    BORDER_DURATION: 5,
    TYPE_BATCHNORM: constants.BN,
    TYPE_DROPOUT: constants.SEQUENCE_DROP,
    DROP_RATE_BEFORE_LSTM: 0.3,
    DROP_RATE_HIDDEN: 0.5,
    DROP_RATE_OUTPUT: 0.0,
    FB_LIST: [1.0],
    TRAINABLE_WAVELET: True,
    WAVELET_SIZE_FACTOR: 1.5,
    USE_LOG: False,
    N_SCALES: 48,
    LOWER_FREQ: 0.5,
    UPPER_FREQ: 30,
    USE_RELU: True,
    INITIAL_LSTM_UNITS: 256,
    INITIAL_CONV_FILTERS: 16,
    INITIAL_KERNEL_SIZE: 5,
    CONV_DOWNSAMPLING: constants.AVGPOOL,
    FC_UNITS: 128,
    CLASS_WEIGHTS: None,
    TYPE_LOSS: constants.CROSS_ENTROPY_LOSS,
    LEARNING_RATE: 1e-4,
    CLIP_NORM: 1,
    MOMENTUM: 0.9,
    USE_NESTEROV_MOMENTUM: False,
    TYPE_OPTIMIZER: constants.ADAM_OPTIMIZER,
    MAX_ITERS: 30000,
    ITERS_STATS: 100,
    ITERS_LR_UPDATE: 1000,
    REL_TOL_CRITERION: 0.0,
    LR_UPDATE_FACTOR: 0.5,
    LR_UPDATE_CRITERION: constants.LOSS_CRITERION,
    MAX_LR_UPDATES: 4,
    TOTAL_DOWNSAMPLING_FACTOR: 8,
    SS_MIN_SEPARATION: 0.3,
    SS_MIN_DURATION: 0.2,
    SS_MAX_DURATION: 4.0,
    KC_MIN_SEPARATION: None,
    KC_MIN_DURATION: 0.3,
    KC_MAX_DURATION: None,
    PREDICT_WITH_AUGMENTED_PAGE: True
}
