from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import datetime

from cwt_layer import complex_morlet_layer
from context_net import context_net

class SpindleDetectorLSTM(object):

    def __init__(self, params):
        pass