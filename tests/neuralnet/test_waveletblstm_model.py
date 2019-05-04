from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

detector_path = '../../..'
sys.path.append(detector_path)

from sleep.neuralnet.models import WaveletBLSTM
from sleep.common import constants
from sleep.common import pkeys


if __name__ == '__main__':
    # Parameters
    params = pkeys.default_params.copy()
    params[pkeys.MODEL_VERSION] = constants.DUMMY
    model = WaveletBLSTM(params)
