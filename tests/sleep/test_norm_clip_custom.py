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
    custom_scaling_dict = {
        # non-test subjects
        1: 1.01,
        3: 10.03,
        5: 1.05,
        7: 1.07,
        9: 1.09,
        10: 1.010,
        11: 1.011,
        14: 1.014,
        17: 1.017,
        18: 1.018,
        19: 1.019,
        # Test subjects
        2: 1.0,
        6: 1.0,
        12: 1.0,
        13: 1.0
    }

    params = {
        pkeys.NORM_COMPUTATION_MODE: constants.NORM_GLOBAL_CUSTOM,
    }
    dataset_name = constants.MASS_SS_NAME
    dataset = load_dataset(dataset_name, params=params, custom_scaling_dict=custom_scaling_dict)
    signal = dataset.get_subject_signal(
        3, normalize_clip=True, normalization_mode=constants.N2_RECORD)
    plt.hist(signal)
    plt.show()
