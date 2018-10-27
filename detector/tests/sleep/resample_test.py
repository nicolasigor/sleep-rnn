from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

detector_path = '../../'
sys.path.append(detector_path)

from sleep.data_ops import resample_eeg

PATH_SAMPLE = 'demo_signal.csv'

if __name__ == '__main__':
    demo_signal = np.loadtxt("demo_signal.csv")
    fs = 200
