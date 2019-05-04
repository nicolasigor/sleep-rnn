from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np

detector_path = '../../..'
sys.path.append(detector_path)

from sleep.data.utils import broad_filter, power_spectrum

PATH_SAMPLE = '../demo_data/demo_signal.csv'

if __name__ == '__main__':
    demo_signal = np.loadtxt(PATH_SAMPLE)
    fs = 200
    filtered_signal = broad_filter(demo_signal, fs, lowcut=0.5, highcut=35)
    time_axis = np.arange(demo_signal.size) / fs
    power, freq = power_spectrum(demo_signal, fs)
    filt_power, _ = power_spectrum(filtered_signal, fs)


    # Show results
    fig, ax = plt.subplots(2, 1)
    # Time-domain signals
    ax[0].plot(time_axis, demo_signal, label='Original')
    ax[0].plot(time_axis, filtered_signal, label='Filtered')
    ax[0].set_title('Filter Test')
    ax[0].set_xlabel('Time [s]')
    ax[0].legend()
    # Power spectrum
    ax[1].plot(freq, power, label='Original')
    ax[1].plot(freq, filt_power, label='Filtered')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_title('Spectrum')
    ax[1].legend()

    plt.show()
