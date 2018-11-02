"""spline.py: Module that implements the CWT using learnable spline wavelets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .cwt_ops import apply_wavelets
from utils.constants import CHANNELS_LAST

# TODO: implement spline filters here
