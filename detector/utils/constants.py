"""constants.py: Module that stores several useful constants for the project."""

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

# Error message, parameter not in valid list
ERROR_INVALID = 'Expected %s for %s, but %s was provided.'
