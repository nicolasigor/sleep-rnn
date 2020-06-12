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
CLIP_VALUE = 'clip_value'
NORM_COMPUTATION_MODE = 'norm_computation_mode'
BATCH_SIZE = 'batch_size'
SHUFFLE_BUFFER_SIZE = 'shuffle_buffer_size'
PREFETCH_BUFFER_SIZE = 'prefetch_buffer_size'
PAGE_DURATION = 'page_duration'
AUG_RESCALE_NORMAL_PROBA = 'aug_rescale_normal_proba'
AUG_RESCALE_NORMAL_STD = 'aug_rescale_normal_std'
AUG_GAUSSIAN_NOISE_PROBA = 'aug_gaussian_noise_proba'
AUG_GAUSSIAN_NOISE_STD = 'aug_gaussian_noise_std'
AUG_RESCALE_UNIFORM_PROBA = 'aug_rescale_uniform_proba'
AUG_RESCALE_UNIFORM_INTENSITY = 'aug_rescale_uniform_intensity'
AUG_ELASTIC_PROBA = 'aug_elastic_proba'
AUG_ELASTIC_ALPHA = 'aug_elastic_alpha'
AUG_ELASTIC_SIGMA = 'aug_elastic_sigma'

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
OUTPUT_LSTM_UNITS = 'output_lstm_units'
# Time-domain convolutional params
TIME_CONV_FILTERS_1 = 'time_conv_filters_1'
TIME_CONV_FILTERS_2 = 'time_conv_filters_2'
TIME_CONV_FILTERS_3 = 'time_conv_filters_3'
SIGMA_FILTER_NTAPS = 'sigma_filter_ntaps'
# cwt domain convolutional params
CWT_CONV_FILTERS_1 = 'cwt_conv_filters_1'
CWT_CONV_FILTERS_2 = 'cwt_conv_filters_2'
CWT_CONV_FILTERS_3 = 'cwt_conv_filters_3'
# General cwt
CWT_RETURN_REAL_PART = 'cwt_return_real_part'
CWT_RETURN_IMAG_PART = 'cwt_return_imag_part'
CWT_RETURN_MAGNITUDE = 'cwt_return_magnitude'
CWT_RETURN_PHASE = 'cwt_return_phase'
INIT_POSITIVE_PROBA = 'init_positive_proba'
# Upconv output
LAST_OUTPUT_CONV_FILTERS = 'last_output_conv_filters'
# UNET parameters
UNET_TIME_INITIAL_CONV_FILTERS = 'unet_time_initial_conv_filters'
UNET_TIME_LSTM_UNITS = 'unet_time_lstm_units'
UNET_TIME_N_DOWN = 'unet_time_n_down'
UNET_TIME_N_CONV_DOWN = 'unet_time_n_conv_down'
UNET_TIME_N_CONV_UP = 'unet_time_n_conv_up'
# Attention parameters
ATT_DIM = 'att_dim'
ATT_N_HEADS = 'att_n_heads'
ATT_PE_FACTOR = 'att_pe_factor'
ATT_DROP_RATE = 'att_drop_rate'
ATT_LSTM_DIM = 'att_lstm_dim'
ATT_PE_CONCAT_DIM = 'att_pe_concat_dim'
ABLATION_TYPE_BATCHNORM_INPUT = 'ablation_type_batchnorm_input'
ABLATION_TYPE_BATCHNORM_CONV = 'ablation_type_batchnorm_conv'
ABLATION_DROP_RATE = 'ablation_drop_rate'
OUTPUT_RESIDUAL_FC_SIZE = 'output_residual_fc_size'
OUTPUT_USE_BN = 'output_use_bn'
OUTPUT_USE_DROP = 'output_use_drop'
FC_UNITS_1 = 'fc_units_1'
FC_UNITS_2 = 'fc_units_2'
SHIELD_LSTM_DOWN_FACTOR = 'shield_lstm_down_factor'
SHIELD_LSTM_TYPE_POOL = 'shield_lstm_type_pool'


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
FOCUSING_PARAMETER = 'focusing_parameter'
WORST_MINING_MIN_NEGATIVE = 'worst_mining_min_negative'
WORST_MINING_FACTOR_NEGATIVE = 'worst_mining_factor_negative'
NEG_ENTROPY_PARAMETER = 'neg_entropy_parameter'
SOFT_LABEL_PARAMETER = 'soft_label_parameter'
MIS_WEIGHT_PARAMETER = 'mis_weight_parameter'
BORDER_WEIGHT_AMPLITUDE = 'border_weight_amplitude'
BORDER_WEIGHT_HALF_WIDTH = 'border_weight_half_width'
MIX_WEIGHTS_STRATEGY = 'mix_weights_strategy'
PREDICTION_VARIABILITY_REGULARIZER = 'prediction_variability_regularizer'
PREDICTION_VARIABILITY_LAG = 'prediction_variability_lag'


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
WEIGHT_DECAY_FACTOR = 'weight_decay_factor'


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
FACTOR_INIT_LR_FINE_TUNE = 'factor_init_lr_fine_tune'
LR_UPDATE_RESET_OPTIMIZER = 'lr_update_reset_optimizer'
KEEP_BEST_VALIDATION = 'keep_best_validation'
FORCED_SEPARATION_DURATION = 'forced_separation_duration'


""" Postprocessing params 
"""
TOTAL_DOWNSAMPLING_FACTOR = 'total_downsampling_factor'
ALIGNED_DOWNSAMPLING = 'aligned_downsampling'
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
    CLIP_VALUE: 10,
    NORM_COMPUTATION_MODE: constants.NORM_GLOBAL,
    BATCH_SIZE: 32,
    SHUFFLE_BUFFER_SIZE: 10000,
    PREFETCH_BUFFER_SIZE: 2,
    PAGE_DURATION: 20,
    AUG_RESCALE_NORMAL_PROBA: 0.0,
    AUG_GAUSSIAN_NOISE_PROBA: 0.0,
    AUG_RESCALE_UNIFORM_PROBA: 0.0,
    AUG_ELASTIC_PROBA: 0.0,
    AUG_RESCALE_NORMAL_STD: 0.05,
    AUG_GAUSSIAN_NOISE_STD: 0.01,
    AUG_RESCALE_UNIFORM_INTENSITY: 0.1,
    AUG_ELASTIC_ALPHA: 0.2,
    AUG_ELASTIC_SIGMA: 0.1,
    MODEL_VERSION: constants.V21,
    BORDER_DURATION: 5,
    TYPE_BATCHNORM: constants.BN,
    TYPE_DROPOUT: constants.SEQUENCE_DROP,
    DROP_RATE_BEFORE_LSTM: 0.2,
    DROP_RATE_HIDDEN: 0.5,
    DROP_RATE_OUTPUT: 0.0,
    FB_LIST: [0.5],
    TRAINABLE_WAVELET: True,
    WAVELET_SIZE_FACTOR: 1.5,
    USE_LOG: None,
    N_SCALES: 32,
    LOWER_FREQ: 0.5,
    UPPER_FREQ: 30,
    USE_RELU: None,
    INITIAL_LSTM_UNITS: 256,
    INITIAL_CONV_FILTERS: None,
    CWT_CONV_FILTERS_1: 32,
    CWT_CONV_FILTERS_2: 64,
    CWT_CONV_FILTERS_3: None,
    INITIAL_KERNEL_SIZE: 3,
    CONV_DOWNSAMPLING: constants.AVGPOOL,
    FC_UNITS: 128,
    OUTPUT_LSTM_UNITS: None,
    LAST_OUTPUT_CONV_FILTERS: None,
    TIME_CONV_FILTERS_1: 64,
    TIME_CONV_FILTERS_2: 128,
    TIME_CONV_FILTERS_3: 256,
    CLASS_WEIGHTS: None,
    TYPE_LOSS: constants.CROSS_ENTROPY_LOSS,
    LEARNING_RATE: 1e-4,
    CLIP_NORM: 1,
    MOMENTUM: 0.9,
    USE_NESTEROV_MOMENTUM: False,
    TYPE_OPTIMIZER: constants.ADAM_OPTIMIZER,
    MAX_ITERS: 30000,
    ITERS_STATS: 50,
    ITERS_LR_UPDATE: 1000,
    REL_TOL_CRITERION: 0.0,
    LR_UPDATE_FACTOR: 0.5,
    LR_UPDATE_CRITERION: constants.LOSS_CRITERION,
    MAX_LR_UPDATES: 4,
    LR_UPDATE_RESET_OPTIMIZER: True,
    TOTAL_DOWNSAMPLING_FACTOR: 8,
    SS_MIN_SEPARATION: 0.3,
    SS_MIN_DURATION: 0.3,
    SS_MAX_DURATION: 3.0,
    KC_MIN_SEPARATION: None,
    KC_MIN_DURATION: 0.3,
    KC_MAX_DURATION: None,
    PREDICT_WITH_AUGMENTED_PAGE: True,
    FACTOR_INIT_LR_FINE_TUNE: 1.0,
    SIGMA_FILTER_NTAPS: None,
    CWT_RETURN_REAL_PART: True,
    CWT_RETURN_IMAG_PART: True,
    CWT_RETURN_MAGNITUDE: False,
    CWT_RETURN_PHASE: False,
    WORST_MINING_MIN_NEGATIVE: None,
    WORST_MINING_FACTOR_NEGATIVE: None,
    KEEP_BEST_VALIDATION: False,
    UNET_TIME_INITIAL_CONV_FILTERS: None,
    UNET_TIME_LSTM_UNITS: None,
    UNET_TIME_N_DOWN: None,
    UNET_TIME_N_CONV_DOWN: None,
    UNET_TIME_N_CONV_UP: None,
    FORCED_SEPARATION_DURATION: 0,
    ALIGNED_DOWNSAMPLING: True,
    ATT_DIM: 512,
    ATT_N_HEADS: 8,
    ATT_PE_FACTOR: 10000,
    ATT_DROP_RATE: 0.1,
    ATT_LSTM_DIM: 256,
    ATT_PE_CONCAT_DIM: 256,
    INIT_POSITIVE_PROBA: 0.1,
    FOCUSING_PARAMETER: None,
    NEG_ENTROPY_PARAMETER: None,
    SOFT_LABEL_PARAMETER: None,
    ABLATION_TYPE_BATCHNORM_INPUT: None,
    ABLATION_TYPE_BATCHNORM_CONV: None,
    ABLATION_DROP_RATE: None,
    MIS_WEIGHT_PARAMETER: None,
    OUTPUT_RESIDUAL_FC_SIZE: None,
    OUTPUT_USE_BN: None,
    OUTPUT_USE_DROP: None,
    FC_UNITS_1: None,
    FC_UNITS_2: None,
    SHIELD_LSTM_DOWN_FACTOR: None,
    SHIELD_LSTM_TYPE_POOL: None,
    BORDER_WEIGHT_AMPLITUDE: None,
    BORDER_WEIGHT_HALF_WIDTH: None,
    MIX_WEIGHTS_STRATEGY: None,
    PREDICTION_VARIABILITY_REGULARIZER: None,
    WEIGHT_DECAY_FACTOR: None,
    PREDICTION_VARIABILITY_LAG: 1
}
