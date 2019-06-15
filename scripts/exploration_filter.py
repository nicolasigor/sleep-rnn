from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from scipy.signal import firwin, firwin2, lfilter, filtfilt
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.data.utils import narrow_filter


def filter_experimental(signal, lowcut, highcut, fs):
    ntaps = 41
    cutoff = [lowcut, highcut]

    n_points = 512
    cutoff_norm = np.asarray(cutoff) / (fs/2)
    freq_array = np.arange(n_points + 1) / n_points

    gain = np.zeros(freq_array.shape)
    gain[1:] = 1 / (1 + np.exp(- 100 * (freq_array[1:] - cutoff_norm[0])))
    gain[1:] -= 1 / (1 + np.exp(- 100 * (freq_array[1:] - cutoff_norm[1])))
    b_alt = firwin2(ntaps, freq_array, gain)
    # filtered_alternative = lfilter(b_alt, [1.0], signal)
    filtered_alternative = filtfilt(b_alt, [1.0], signal)

    # Remove borders
    filtered_signal = np.zeros(filtered_alternative.shape)
    crop_size = (ntaps-1)//2
    filtered_signal[crop_size:-crop_size] = filtered_alternative[crop_size:-crop_size]

    # shift
    # filtered_signal = np.zeros(filtered_alternative.shape)
    # filtered_signal[:-(ntaps-1)//2] = filtered_alternative[(ntaps-1)//2:]

    return filtered_signal


if __name__ == '__main__':

    dataset = load_dataset('mass_ss')
    fs = dataset.fs
    x = dataset.get_subject_signal(subject_id=1, normalize_clip=False)
    y = dataset.get_subject_stamps(subject_id=1)
    which_stamp = 337
    center_sample = int(y[which_stamp, :].mean())
    start_sample = center_sample - 5 * fs
    end_sample = center_sample + 5 * fs
    signal = x[start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / fs
    useful_stamps = np.where(
        (y[:, 1] >= start_sample) & (y[:, 0] <= end_sample)
    )[0]

    bands_to_show = [
        (0.5, 4),
        (4, 8),
        (8, 15),
        (15, 30),
        (12, 14)
    ]
    fig, ax = plt.subplots(1 + len(bands_to_show), 1, figsize=(8, 6), dpi=100)

    ax[0].plot(time_axis, signal)
    for to_show_stamp in useful_stamps:
        ax[0].fill_between(y[to_show_stamp, :]/fs, 25, -25, alpha=0.3)
    ax[0].set_xlim([time_axis[0], time_axis[-1]])

    for k, band in enumerate(bands_to_show):

        # Filter by Rosario
        filtered_signal = filter_experimental(
            signal, lowcut=band[0], highcut=band[1], fs=fs)

        ax[k + 1].plot(time_axis, filtered_signal, label='%s-%s Hz' % band)
        ax[k + 1].set_xlim([time_axis[0], time_axis[-1]])
        ax[k + 1].legend(loc='upper right', fontsize=8)
    plt.show()