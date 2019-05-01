"""constants.py: Module that stores several useful constants for the project."""

# Dataset name
MASS_SS_NAME = 'mass_ss'
INTA_SS_NAME = 'inta_ss'
MASS_KC_NAME = 'mass_kc'

# Database split
TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'
TEST_SUBSET = 'test'

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

# Types of loss
CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
DICE_LOSS = 'dice_loss'

# Types of optimizer
ADAM_OPTIMIZER = 'adam_optimizer'
SGD_OPTIMIZER = 'sgd_optimizer'
RMSPROP_OPTIMIZER = 'rmsprop_optimizer'

# Training params
LOSS_CRITERION = 'loss_criterion'
METRIC_CRITERION = 'metric_criterion'

# Model versions
DUMMY = 'dummy'
V1 = 'v1'
V2 = 'v2'
V3 = 'v3'
V3_FF = 'v3-ff'
V3_CONV = 'v3-conv'
V3_FF_CONV = 'v3-ff-conv'
EXPERIMENTAL = 'experimental'
V4 = 'v4'
