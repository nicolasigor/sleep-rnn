from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from scipy.signal import firwin, firwin2, lfilter, filtfilt
from scipy import signal as spsignal
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, sproot

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset

CUSTOM_COLOR = {
    'red': '#c62828', 'grey': '#455a64', 'blue': '#0277bd', 'green': '#43a047'}


class MultiplePeaks(Exception):
    pass


class NoPeaksFound(Exception):
    pass


def get_central_crop(signal, crop_size):
    center_sample = signal.size//2
    start_sample = int(center_sample - crop_size // 2)
    end_sample = int(center_sample + crop_size // 2 + 1)
    return signal[start_sample:end_sample]


def gaussian_window(ntaps):
    return spsignal.gaussian(ntaps, std=ntaps/8)


def power_spectrum(signal, fs):
    """Returns the single-sided power spectrum of the signal using FFT"""
    n = signal.size
    y = np.fft.fft(signal)
    y = np.abs(y) / n
    power = y[:n // 2]
    power[1:-1] = 2 * power[1:-1]
    freq = np.fft.fftfreq(n, d=1 / fs)
    freq = freq[:n // 2]
    return power, freq


def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.max(y) / 2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return roots


def filter_windowed_sinusoidal(
        signal, window_fn, fs, central_freq, ntaps, sinusoidal_fn=np.cos):
    # Kernel design
    time_array = np.arange(ntaps) - ntaps // 2
    time_array = time_array / fs
    b_base = sinusoidal_fn(2 * np.pi * central_freq * time_array)
    cos_base = np.cos(2 * np.pi * central_freq * time_array)
    window = window_fn(b_base.size)
    norm_factor = np.sum(window * (cos_base ** 2))
    kernel = b_base * window / norm_factor

    # Apply kernel
    filtered = lfilter(kernel, [1.0], signal)
    # Shift
    ntaps = kernel.size
    filtered_signal = np.zeros(filtered.shape)
    filtered_signal[:-(ntaps - 1) // 2] = filtered[(ntaps - 1) // 2:]
    return filtered_signal, kernel


def filter_sigma_rosario(signal, fs, ntaps=21):
    width = 0.5
    cutoff = [12, 14]

    # Kernel design
    b_base = firwin(ntaps, cutoff, width, pass_zero=False, fs=fs)
    kernel = np.append(np.array([0, 0]), b_base) - np.append(
        b_base, np.array([0, 0]))

    # Normalize kernel
    kernel = kernel / np.linalg.norm(kernel)

    # Apply kernel
    filtered = lfilter(kernel, [1.0], signal)
    # Shift
    ntaps = kernel.size
    filtered_signal = np.zeros(filtered.shape)
    filtered_signal[:-(ntaps - 1) // 2] = filtered[(ntaps - 1) // 2:]
    return filtered_signal, kernel


if __name__ == '__main__':

    dataset = load_dataset('mass_ss')
    fs = dataset.fs
    x = dataset.get_subject_signal(subject_id=2, normalize_clip=False)
    y = dataset.get_subject_stamps(subject_id=2)
    which_stamp = 425  # 337  # 425
    context_size = 6
    center_sample = int(y[which_stamp, :].mean())
    start_sample = center_sample - (context_size//2) * fs
    end_sample = center_sample + (context_size//2) * fs
    signal = x[start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / fs
    useful_stamps = np.where(
        (y[:, 1] >= start_sample) & (y[:, 0] <= end_sample))[0]
    time_axis = time_axis - start_sample / fs

    # ----------------- PLOT
    fig, ax = plt.subplots(4, 1, figsize=(6, 6), dpi=100)
    title_font = 9
    other_font = 7

    # display_stamps = y[useful_stamps] - start_sample
    # np.save('demo_mark.npy', display_stamps)


    # Signal
    ax[0].set_title(
        'Sleep Spindle, C3-CLE EEG Channel',
        fontsize=title_font)
    ax[0].plot(time_axis, signal, linewidth=1, color=CUSTOM_COLOR['grey'])
    for to_show_stamp in useful_stamps:
        ax[0].fill_between((y[to_show_stamp, :] - start_sample)/fs, 50, -50,
                           alpha=0.4, facecolor=CUSTOM_COLOR['blue'])
    ax[0].set_xlim([time_axis[0], time_axis[-1]])
    ax[0].set_yticks([])
    ax[0].tick_params(labelsize=other_font)
    ax[0].set_xlabel('Time [s]', fontsize=other_font)

    # Filter Dict
    filters_dict = {
        # 'Rosario': filter_sigma_rosario(signal, fs, ntaps=21),
        # 'Rect': filter_windowed_sinusoidal(signal, lambda x: np.kaiser(x, beta=0), fs, 13, 43),
         #'Hamming': filter_windowed_sinusoidal(signal, np.hamming, fs, 13, 61),
        'Sigma': filter_windowed_sinusoidal(signal, np.hanning, fs, 13, 51, sinusoidal_fn=np.cos),
        'Below': filter_windowed_sinusoidal(signal, np.hanning, fs, 6, 71, sinusoidal_fn=np.cos),
        # 'Cos+Hanning 2': filter_windowed_sinusoidal(signal, np.hanning, fs, 8, 61, sinusoidal_fn=np.cos),
        # 'Bartlett': filter_windowed_sinusoidal(signal, np.bartlett, fs, 13, 61),
        # 'Blackman': filter_windowed_sinusoidal(signal, np.blackman, fs, 13, 61),
        #'Kaiser': filter_windowed_sinusoidal(signal, lambda x: np.kaiser(x, beta=4), fs, 13, 41),
        #'Gaussian': filter_windowed_sinusoidal(signal, gaussian_window, fs, 13, 51),
    }

    # kernel stuff
    kernel_sizes = []
    for key in filters_dict:
        kernel_sizes.append(filters_dict[key][1].size)
    max_kernel_size = np.max(kernel_sizes)
    max_kernel_size_to_show = max_kernel_size + 10
    max_kernel_size_to_fft = max_kernel_size + 2000
    kernel_base = np.zeros(max_kernel_size_to_fft)
    half_sample = kernel_base.size//2
    kernel_axis = np.arange(max_kernel_size_to_show) - max_kernel_size_to_show//2

    max_freq = 40

    for key in filters_dict:
        filtered_signal, kernel = filters_dict[key]

        ax[1].plot(
            time_axis, filtered_signal,
            label='Result with Filter %s' % key, linewidth=1)

        kernel_full = kernel_base.copy()
        kernel_half_size = kernel.size // 2
        start_kernel = half_sample - kernel_half_size
        end_kernel = half_sample + kernel_half_size + 1
        kernel_full[start_kernel:end_kernel] = kernel

        ax[2].plot(
            kernel_axis,
            get_central_crop(kernel_full, max_kernel_size_to_show),
            label='Filter %s' % key, linewidth=1)
        # ax[2].plot(
        #     kernel_axis[start_kernel:end_kernel],
        #     get_central_crop(kernel_full, max_kernel_size_to_show)[start_kernel:end_kernel],
        #     marker='.')

        fft_kernel, freq_axis = power_spectrum(kernel_full, fs)
        fft_kernel = fft_kernel / fft_kernel.max()
        fft_kernel = fft_kernel[freq_axis <= max_freq]
        freq_axis = freq_axis[freq_axis <= max_freq]

        response = fft_kernel  # 20 * np.log10(np.abs(fft_kernel))

        kernel_fwhm = fwhm(freq_axis, fft_kernel**2)
        ax[3].plot(
            freq_axis, response, #label='%s' % key,
            label='%s [%1.1f %1.1f] Hz' % (
                key, kernel_fwhm[0], kernel_fwhm[1]),
            linewidth=1)

    ax[1].set_title(
        'Filtered signal',
        fontsize=title_font)
    ax[1].set_xlim([time_axis[0], time_axis[-1]])
    ax[1].legend(loc='upper right', fontsize=other_font)
    ax[1].set_yticks([])
    ax[1].tick_params(labelsize=other_font)
    ax[1].set_xlabel('Time [s]', fontsize=other_font)

    ax[2].set_title(
        'Filter',
        fontsize=title_font)
    ax[2].set_xlim([kernel_axis[0], kernel_axis[-1]])
    ax[2].legend(loc='upper right', fontsize=other_font)
    ax[2].set_yticks([])
    ax[2].tick_params(labelsize=other_font)
    ax[2].set_xlabel('Taps', fontsize=other_font)

    ax[3].set_title(
        'Frequency response of filter',
        fontsize=title_font)
    ax[3].set_xlim([0, 40])
    ax[3].legend(loc='upper right', fontsize=other_font)
    #ax[3].set_yticks([])
    ax[3].tick_params(labelsize=other_font)
    ax[3].set_xlabel('Frequency [Hz]', fontsize=other_font)

    plt.tight_layout()
    plt.show()
