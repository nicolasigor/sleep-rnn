"""models.py: Module that defines neural network
models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import layers

from utils.constants import CHANNELS_LAST, CHANNELS_FIRST
from utils.constants import PAD_SAME, PAD_VALID
from utils.constants import BN, BN_RENORM
from utils.constants import MAXPOOL, AVGPOOL
from utils.constants import SEQUENCE_DROP, REGULAR_DROP
from utils.constants import UNIDIRECTIONAL, BIDIRECTIONAL
from utils.constants import ERROR_INVALID

