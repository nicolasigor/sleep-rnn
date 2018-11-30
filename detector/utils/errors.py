"""Module that defines common errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import constants


def check_valid_value(value, name, valid_list):
    """Raises a ValueError exception if value not in valid_list"""
    if value not in valid_list:
        msg = constants.ERROR_INVALID \
              % (valid_list, name, value)
        raise ValueError(msg)
