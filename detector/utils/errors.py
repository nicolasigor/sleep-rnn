"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import constants


def check_valid_value(value, name, valid_list):
    """Raises a ValueError exception if value not in valid_list"""
    if value not in valid_list:
        msg = constants.ERROR_INVALID \
              % (valid_list, name, value)
        raise ValueError(msg)

