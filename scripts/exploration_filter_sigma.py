from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from scipy.signal import firwin, firwin2, lfilter, filtfilt
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, sproot

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.data.utils import power_spectrum

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


def filter_sigma_alt_1(signal, fs):
    ntaps = 41
    cutoff = [12, 14]

    # Kernel design
    n_points = 512
    cutoff_norm = np.asarray(cutoff) / (fs/2)
    freq_array = np.arange(n_points + 1) / n_points
    gain = np.zeros(freq_array.shape)
    gain[1:] = 1 / (1 + np.exp(- 100 * (freq_array[1:] - cutoff_norm[0])))
    gain[1:] -= 1 / (1 + np.exp(- 100 * (freq_array[1:] - cutoff_norm[1])))
    kernel = firwin2(ntaps, freq_array, gain)

    # Normalize kernel
    kernel = kernel / np.linalg.norm(kernel)

    # Apply kernel
    filtered = lfilter(kernel, [1.0], signal)
    # Shift
    ntaps = kernel.size
    filtered_signal = np.zeros(filtered.shape)
    filtered_signal[:-(ntaps - 1) // 2] = filtered[(ntaps - 1) // 2:]
    return filtered_signal, kernel


def filter_sigma_alt_2(signal, fs):
    ntaps = 41
    cutoff = [12, 14]

    # Kernel design
    time_array = np.arange(ntaps) - ntaps//2
    time_array = time_array / fs
    b_base = np.cos(2*np.pi*np.mean(cutoff)*time_array)
    kernel = b_base * np.hamming(b_base.size)

    # Normalize kernel
    kernel = kernel / np.linalg.norm(kernel)

    # Apply kernel
    filtered = lfilter(kernel, [1.0], signal)
    # Shift
    ntaps = kernel.size
    filtered_signal = np.zeros(filtered.shape)
    filtered_signal[:-(ntaps - 1) // 2] = filtered[(ntaps - 1) // 2:]
    return filtered_signal, kernel


def filter_sigma_rosario(signal, fs):
    ntaps = 21
    width = 0.5
    cutoff = [12, 14]

    # Kernel design
    b_base = firwin(ntaps, cutoff, width, pass_zero=False, fs=fs)
    kernel = np.append(np.array([0, 0]), b_base) - np.append(
        b_base, np.array([0, 0]))
    # kernel = b_base
    kernel = np.hamming(kernel.size) * kernel
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
    which_stamp = 425  # 337
    center_sample = int(y[which_stamp, :].mean())
    start_sample = center_sample - 3 * fs
    end_sample = center_sample + 3 * fs
    signal = x[start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / fs
    useful_stamps = np.where(
        (y[:, 1] >= start_sample) & (y[:, 0] <= end_sample)
    )[0]

    time_axis = time_axis - start_sample / fs
    fig, ax = plt.subplots(4, 1, figsize=(8, 6), dpi=200)
    title_font = 9
    other_font = 7

    # Signal
    ax[0].set_title(
        'Sleep Spindle, C3-CLE EEG Channel (Time in seconds)',
        fontsize=title_font)
    ax[0].plot(time_axis, signal, linewidth=1.5, color=CUSTOM_COLOR['grey'])
    for to_show_stamp in useful_stamps:
        ax[0].fill_between((y[to_show_stamp, :] - start_sample)/fs, 50, -50,
                           alpha=0.4, facecolor=CUSTOM_COLOR['blue'])
    ax[0].set_xlim([time_axis[0], time_axis[-1]])
    ax[0].set_yticks([])
    ax[0].tick_params(labelsize=other_font)

    # Filter by Rosario
    filtered_rosario, kernel_rosario = filter_sigma_rosario(signal, fs=fs)
    ax[1].plot(time_axis, filtered_rosario, label='%s-%s Hz Rosario' % (12, 14),
               linewidth=1, color=CUSTOM_COLOR['red'])
    # Filter Alt 1
    filtered_alt_1, kernel_alt_1 = filter_sigma_alt_1(signal, fs=fs)
    # ax[1].plot(time_axis, filtered_alt_1, label='%s-%s Hz Alt1' % (12, 14),
    #            linewidth=1, color=CUSTOM_COLOR['blue'])
    # Filter Alt 2
    filtered_alt_2, kernel_alt_2 = filter_sigma_alt_2(signal, fs=fs)
    ax[1].plot(time_axis, filtered_alt_2, label='%s-%s Hz Alt2' % (12, 14),
               linewidth=1, color=CUSTOM_COLOR['green'])

    ax[1].set_xlim([time_axis[0], time_axis[-1]])
    ax[1].legend(loc='upper right', fontsize=other_font)
    ax[1].set_yticks([])
    ax[1].tick_params(labelsize=other_font)

    # Kernels
    max_kernel_size_original = np.max([
        kernel_rosario.size,
        kernel_alt_1.size,
        kernel_alt_2.size
    ])
    max_kernel_size_to_show = max_kernel_size_original + 10
    max_kernel_size = max_kernel_size_original + 200
    kernel_axis = np.arange(max_kernel_size)
    kernel_base = np.zeros(kernel_axis.shape)
    half_sample = kernel_base.size//2

    kernel_full = kernel_base.copy()
    kernel_half_size = kernel_rosario.size//2
    start_kernel = half_sample - kernel_half_size
    end_kernel = half_sample + kernel_half_size + 1
    kernel_full[start_kernel:end_kernel] = kernel_rosario
    fft_kernel_rosario, _ = power_spectrum(kernel_full, fs)

    ax[2].plot(
        get_central_crop(kernel_axis, max_kernel_size_to_show),
        get_central_crop(kernel_full, max_kernel_size_to_show),
        label='Kernel Rosario',
        linewidth=1, color=CUSTOM_COLOR['red'], marker='.')

    kernel_full = kernel_base.copy()
    kernel_half_size = kernel_alt_1.size // 2
    start_kernel = half_sample - kernel_half_size
    end_kernel = half_sample + kernel_half_size + 1
    kernel_full[start_kernel:end_kernel] = kernel_alt_1
    fft_kernel_alt_1, _ = power_spectrum(kernel_full, fs)
    ax[2].plot(
        get_central_crop(kernel_axis, max_kernel_size_to_show),
        get_central_crop(kernel_full, max_kernel_size_to_show),
        label='Kernel Alt 1',
        linewidth=1, color=CUSTOM_COLOR['blue'], marker='.')

    kernel_full = kernel_base.copy()
    kernel_half_size = kernel_alt_2.size // 2
    start_kernel = half_sample - kernel_half_size
    end_kernel = half_sample + kernel_half_size + 1
    kernel_full[start_kernel:end_kernel] = kernel_alt_2
    fft_kernel_alt_2, freq_axis = power_spectrum(kernel_full, fs)
    ax[2].plot(
        get_central_crop(kernel_axis, max_kernel_size_to_show),
        get_central_crop(kernel_full, max_kernel_size_to_show),
        label='Kernel Alt 2',
        linewidth=1, color=CUSTOM_COLOR['green'], marker='.')

    ax[2].legend(loc='upper right', fontsize=other_font)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    ax[2].tick_params(labelsize=other_font)

    # Frequency response of kernel
    max_fft_value = np.max([
        fft_kernel_rosario.max(),
        fft_kernel_alt_1.max(),
        fft_kernel_alt_2.max()
    ])

    fwhm_rosario = fwhm(freq_axis, fft_kernel_rosario)
    ax[3].plot(freq_axis, fft_kernel_rosario,
               label='FFT Rosario FWHM: [%1.1f %1.1f] Hz' % tuple(fwhm_rosario),
               linewidth=1, color=CUSTOM_COLOR['red'])

    fwhm_alt_1 = fwhm(freq_axis, fft_kernel_alt_1)
    ax[3].plot(freq_axis, fft_kernel_alt_1,
               label='FFT Alt 1 FWHM: [%1.1f %1.1f] Hz' % tuple(fwhm_alt_1),
               linewidth=1, color=CUSTOM_COLOR['blue'])

    fwhm_alt_2 = fwhm(freq_axis, fft_kernel_alt_2)
    ax[3].plot(freq_axis, fft_kernel_alt_2,
               label='FFT Alt 2 FWHM: [%1.1f %1.1f] Hz' % tuple(fwhm_alt_2),
               linewidth=1, color=CUSTOM_COLOR['green'])


    ax[3].set_xlim([0, 40])
    ax[3].legend(loc='upper right', fontsize=other_font)
    ax[3].set_yticks([])
    #ax[3].set_xticks([])
    ax[3].tick_params(labelsize=other_font)

    plt.tight_layout()
    plt.show()

