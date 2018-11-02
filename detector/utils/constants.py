"""constants.py: Module that stores several useful constants for the project."""

# Data format
CHANNELS_FIRST = 'channels_first'
CHANNELS_LAST = 'channels_last'

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
SEQUENCE_DROP = 'time_dropout'
