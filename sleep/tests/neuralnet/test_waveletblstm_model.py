from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

detector_path = '../../..'
sys.path.append(detector_path)

from sleep.neuralnet.models import WaveletBLSTM
from sleep.utils import constants
from sleep.utils import param_keys


if __name__ == '__main__':
    # Parameters
    params = param_keys.default_params.copy()
    params[param_keys.MODEL_VERSION] = constants.DUMMY
    model = WaveletBLSTM(params)
