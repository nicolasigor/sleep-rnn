"""Module that stores several useful keys to configure a model."""

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
TIME_RESOLUTION_FACTOR = 'time_resolution_factor'
FS = 'fs'
BORDER_DURATION = 'border_duration'
FB_LIST = 'fb_list'
N_CONV_BLOCKS = 'n_conv_blocks'
N_TIME_LEVELS = 'n_time_levels'
BATCHNORM_CONV = 'batchnorm_conv'
POOLING_CONV = 'pooling_conv'
BATCHNORM_FIRST_LSTM = 'batchnorm_first_lstm'
DROPOUT_FIRST_LSTM = 'dropout_first_lstm'
BATCHNORM_REST_LSTM = 'batchnorm_rest_lstm'
DROPOUT_REST_LSTM = 'dropout_rest_lstm'
TIME_POOLING = 'time_pooling'
BATCHNORM_FC = 'batchnorm_fc'
DROPOUT_FC = 'dropout_fc'
DROP_RATE = 'drop_rate'
TRAINABLE_WAVELET = 'trainable_wavelet'
WAVELET_SIZE_FACTOR = 'wavelet_size_factor'
TYPE_WAVELET = 'type_wavelet'
USE_LOG = 'use_log'
N_SCALES = 'n_scales'
LOWER_FREQ = 'lower_freq'
UPPER_FREQ = 'upper_freq'
INITIAL_CONV_FILTERS = 'initial_conv_filters'
INITIAL_LSTM_UNITS = 'initial_lstm_units'
DUPLICATE_AFTER_DOWNSAMPLING_LSTM = 'duplicate_after_downsampling_lstm'

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
clip_gradients: (boolean) Whether to clip the gradient by the global norm.
clip_norm: (float) if clip_gradients is true, this is the global norm to use.
momentum: (float) momentum for the SGD optimizer.
use_nesterov: (bool) whether to use nesterov momentum instead of regular
    momentum for SGD optimization.
type_optimizer: ({ADAM_OPTIMIZER, SGD_OPTIMIZER}) Type of optimizer to be used.
"""
LEARNING_RATE = 'learning_rate'
CLIP_GRADIENTS = 'clip_gradients'
CLIP_NORM = 'clip_norm'
MOMENTUM = 'momentum'
USE_NESTEROV_MOMENTUM = 'use_nesterov'
TYPE_OPTIMIZER = 'type_optimizer'


""" Training params

max_epochs: (int) Maximum numer of epochs to be performed in the training loop.
nstats: (int) Frequency in iterations to display metrics.
"""
MAX_ITERATIONS = 'max_epochs'
NSTATS = 'nstats'


# Default parameters dictionary
default_params = {
    BATCH_SIZE: 32,
    SHUFFLE_BUFFER_SIZE: 1000,
    PREFETCH_BUFFER_SIZE: 2,
    PAGE_DURATION: 20,
    TIME_RESOLUTION_FACTOR: 8,
    FS: 200,
    BORDER_DURATION: 3,
    FB_LIST: [0.5, 1.0, 1.5, 2.0],
    N_CONV_BLOCKS: 0,
    N_TIME_LEVELS: 1,
    BATCHNORM_CONV: constants.BN_RENORM,
    POOLING_CONV: constants.MAXPOOL,
    BATCHNORM_FIRST_LSTM: constants.BN_RENORM,
    DROPOUT_FIRST_LSTM: None,
    BATCHNORM_REST_LSTM: None,
    DROPOUT_REST_LSTM: constants.SEQUENCE_DROP,
    TIME_POOLING: constants.AVGPOOL,
    BATCHNORM_FC: None,
    DROPOUT_FC: constants.SEQUENCE_DROP,
    DROP_RATE: 0.3,
    TRAINABLE_WAVELET: True,
    WAVELET_SIZE_FACTOR: 2.0,
    TYPE_WAVELET: constants.CMORLET,
    USE_LOG: False,
    N_SCALES: 32,
    LOWER_FREQ: 2,
    UPPER_FREQ: 32,
    INITIAL_CONV_FILTERS: 16,
    INITIAL_LSTM_UNITS: 128,
    DUPLICATE_AFTER_DOWNSAMPLING_LSTM: True,
    CLASS_WEIGHTS: None,
    TYPE_LOSS: constants.DICE_LOSS,
    LEARNING_RATE: 0.001,
    CLIP_GRADIENTS: True,
    CLIP_NORM: 5,
    MOMENTUM: 0.9,
    USE_NESTEROV_MOMENTUM: False,
    TYPE_OPTIMIZER: constants.ADAM_OPTIMIZER,
    MAX_ITERATIONS: 15000,
    NSTATS: 50
}
