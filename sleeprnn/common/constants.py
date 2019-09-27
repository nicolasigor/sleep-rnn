"""constants.py: Module that stores several useful constants for the project."""

# Dataset name
MASS_SS_NAME = 'mass_ss'
INTA_SS_NAME = 'inta_ss'
MASS_KC_NAME = 'mass_kc'
DREAMS_SS_NAME = 'dreams_ss'
DREAMS_KC_NAME = 'dreams_kc'

# Database split
TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'
TEST_SUBSET = 'test'

# Task mode
N2_RECORD = 'n2'
WN_RECORD = 'wn'

# Event names
SPINDLE = 'spindle'
KCOMPLEX = 'kcomplex'

# Metric keys
F1_SCORE = 'f1_score'
PRECISION = 'precision'
RECALL = 'recall'
TP = 'tp'
FP = 'fp'
FN = 'fn'
MEAN_ALL_IOU = 'mean_all_iou'
MEAN_NONZERO_IOU = 'mean_nonzero_iou'

# Type of padding
PAD_SAME = 'same'
PAD_VALID = 'valid'

# Type of batch normalization
BN = 'bn'
BN_RENORM = 'bn_renorm'

# Type of pooling
MAXPOOL = 'maxpool'
AVGPOOL = 'avgpool'

# Alternative to pooling after convolution
STRIDEDCONV = 'stridedconv'

# Type of dropout
REGULAR_DROP = 'regular_dropout'
SEQUENCE_DROP = 'sequence_dropout'

# Number of directions for LSTM layers
UNIDIRECTIONAL = 'unidirectional'
BIDIRECTIONAL = 'bidirectional'

# Type of class weights
BALANCED = 'balanced'
BALANCED_DROP = 'balanced_drop'
BALANCED_DROP_V2 = 'balanced_drop_v2'

# Types of loss
CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
DICE_LOSS = 'dice_loss'
FOCAL_LOSS = 'focal_loss'

# Types of optimizer
ADAM_OPTIMIZER = 'adam_optimizer'
SGD_OPTIMIZER = 'sgd_optimizer'
RMSPROP_OPTIMIZER = 'rmsprop_optimizer'

# Training params
LOSS_CRITERION = 'loss_criterion'
METRIC_CRITERION = 'metric_criterion'

# Normalization computation mode
NORM_IQR = 'norm_iqr'
NORM_STD = 'norm_std'
NORM_GLOBAL = 'norm_global'

# Colors
RED = 'red'
BLUE = 'blue'
GREEN = 'green'
GREY = 'grey'
DARK = 'dark'

# Model versions
DUMMY = 'dummy'
DEBUG = 'debug'
V1 = 'v1'
V4 = 'v4'
V5 = 'v5'
V6 = 'v6'
V7 = 'v7'
V8 = 'v8'
V9 = 'v9'
V7lite = 'v7lite'
V7litebig = 'v7litebig'
V10 = 'v10'
V11 = 'v11'  # Time-domain
V12 = 'v12'
V13 = 'v13'
V14 = 'v14'
V15 = 'v15'  # Mixed
V16 = 'v16'  # Mixed
V17 = 'v17'
V18 = 'v18'  # Mixed
V19 = 'v19'
V20_INDEP = 'v20_indep'  # time
V20_CONCAT = 'v20_concat'  # time
V21 = 'v21'  # Mixed
V22 = 'v22'  # CWT indep
V23 = 'v23'  # Time-domain with LSTM instead of FC
