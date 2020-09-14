from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np

detector_path = '../../..'
sys.path.append(detector_path)

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants, pkeys

if __name__ == '__main__':
    params = {
        pkeys.FS: 200,
        pkeys.NORM_COMPUTATION_MODE: constants.NORM_GLOBAL_FFT,
        pkeys.CLIP_VALUE: 10
    }
    dataset_name = constants.MASS_SS_NAME
    dataset = load_dataset(dataset_name, params=params)
    signal = dataset.get_subject_signal(3, normalize_clip=True, normalization_mode=constants.N2_RECORD)
    plt.hist(signal)
    plt.show()
