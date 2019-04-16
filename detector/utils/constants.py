"""constants.py: Module that stores several useful constants for the project."""

# Database name
MASS_NAME = 'mass'
INTA_NAME = 'inta'
MASSK_NAME = 'massk'

# Database split
TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'
TEST_SUBSET = 'test'

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

# Types of wavelet
CMORLET = 'cmorlet'
SPLINE = 'spline'

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

# Error message, parameter not in valid list
ERROR_INVALID = 'Expected %s for %s, but %s was provided.'

# Model versions
DUMMY = 'dummy'
V1 = 'v1'
V2 = 'v2'
V3 = 'v3'
V3_FF = 'v3-ff'
V3_CONV = 'v3-conv'
V3_FF_CONV = 'v3-ff-conv'
EXPERIMENTAL = 'experimental'
