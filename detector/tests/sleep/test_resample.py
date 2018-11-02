from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np

detector_path = '../../'
sys.path.append(detector_path)

from sleep.data_ops import resample_eeg

PATH_SAMPLE = '../demo_data/demo_signal.csv'

if __name__ == '__main__':
    demo_signal = np.loadtxt(PATH_SAMPLE)
    fs = 200

    signal = demo_signal
    fs_old = fs
    fs_new = 100
    resampled_signal = resample_eeg(signal, fs_old, fs_new)

    time_old = np.arange(signal.size) / fs_old
    time_new = np.arange(resampled_signal.size) / fs_new

    fig = plt.figure()
    plt.plot(time_old, signal, label='Original')
    plt.plot(time_new, resampled_signal, label='Resampled')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()
