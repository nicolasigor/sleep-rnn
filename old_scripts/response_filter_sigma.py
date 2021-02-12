from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, sproot


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


def pad_to_size(signal, new_size):
    new_signal = np.zeros(new_size)
    half_sample = new_size // 2
    start_fill = half_sample - signal.size // 2
    end_fill = half_sample + signal.size // 2 + 1
    new_signal[start_fill:end_fill] = signal
    return new_signal


def power_spectrum(signal, fs, n_samples=None, max_frequency=None):
    """Returns the single-sided power spectrum of the signal using FFT"""
    if n_samples is not None:
        if n_samples < signal.size:
            raise ValueError('n_samples is less than signal size.')
        signal = pad_to_size(signal, n_samples)

    n = signal.size
    y = np.fft.fft(signal)
    y = np.abs(y) / n
    power = y[:n // 2]
    power[1:-1] = 2 * power[1:-1]
    freq = np.fft.fftfreq(n, d=1 / fs)
    freq = freq[:n // 2]

    if max_frequency is not None:
        power = power[freq <= max_frequency]
        freq = freq[freq <= max_frequency]

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
    window = window_fn(b_base.size)
    norm_factor = np.sum(window * (b_base ** 2))
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
    time_array = np.arange(kernel.size) - kernel.size // 2
    time_array = time_array / fs
    unity_gain_example = np.sin(
        2 * np.pi * np.mean(cutoff) * time_array)
    kernel = kernel / np.sum(kernel * unity_gain_example)

    # Apply kernel
    filtered = lfilter(kernel, [1.0], signal)
    # Shift
    ntaps = kernel.size
    filtered_signal = np.zeros(filtered.shape)
    filtered_signal[:-(ntaps - 1) // 2] = filtered[(ntaps - 1) // 2:]
    return filtered_signal, kernel


if __name__ == '__main__':

    # Data
    fs = 200
    x = np.load('demo_signal.npy')
    y = np.load('demo_mark.npy')
    time_axis = np.arange(x.size) / fs

    # Filter Dict
    filters_dict = {
        'Rosario': filter_sigma_rosario(
            x, fs, 21),
        'Alt': filter_windowed_sinusoidal(
            x, np.hanning, fs, 13, 41, sinusoidal_fn=np.sin)
    }

    fig, ax = plt.subplots(4, 1, figsize=(6, 6), dpi=100)
    title_font = 9
    other_font = 7

    # Signal
    ax[0].set_title(
        'Sleep Spindle, C3-CLE EEG Channel', fontsize=title_font)
    ax[0].plot(time_axis, x, linewidth=1, color=CUSTOM_COLOR['grey'])
    for single_stamp in y:
        ax[0].fill_between(
            single_stamp / fs, 50, -50,
            alpha=0.4, facecolor=CUSTOM_COLOR['blue'])
    ax[0].set_xlim([time_axis[0], time_axis[-1]])
    ax[0].set_yticks([-50, 0, 50])
    ax[0].set_ylim([-70, 70])
    ax[0].tick_params(labelsize=other_font)
    ax[0].set_xlabel('Time [s]', fontsize=other_font)

    kernel_sizes = [filters_dict[key][1].size for key in filters_dict]
    max_kernel_size = np.max(kernel_sizes)
    size_for_fft = max_kernel_size + 2000
    size_for_display = max_kernel_size + 10
    max_frequency = 40
    kernel_axis = np.arange(size_for_display) - size_for_display//2

    for key in filters_dict:
        filtered_signal, kernel = filters_dict[key]

        ax[1].plot(
            time_axis, filtered_signal,
            label='Result with Filter %s' % key, linewidth=1)

        line = ax[2].plot(
            kernel_axis,
            pad_to_size(kernel, size_for_display),
            label='Filter %s' % key, linewidth=1)
        ax[2].plot(
            get_central_crop(kernel_axis, kernel.size),
            kernel, linestyle='None', marker='.', color=line[0].get_color())

        fft_kernel, freq_axis = power_spectrum(
            kernel, fs, n_samples=size_for_fft, max_frequency=max_frequency)
        fft_kernel = fft_kernel / fft_kernel.max()
        kernel_cutoff = fwhm(freq_axis, fft_kernel**2)

        ax[3].plot(
            freq_axis, fft_kernel,
            label='%s [%1.1f %1.1f] Hz' % (
                key, kernel_cutoff[0], kernel_cutoff[1]),
            linewidth=1)

    ax[1].set_title('Filtered signal', fontsize=title_font)
    ax[1].set_xlim([time_axis[0], time_axis[-1]])
    ax[1].set_ylim([-70, 70])
    ax[1].legend(loc='upper right', fontsize=other_font)
    ax[1].set_yticks([-50, 0, 50])
    ax[1].tick_params(labelsize=other_font)
    ax[1].set_xlabel('Time [s]', fontsize=other_font)

    ax[2].set_title('Filter', fontsize=title_font)
    ax[2].set_xlim([kernel_axis[0], kernel_axis[-1]])
    ax[2].legend(loc='upper right', fontsize=other_font)
    ax[2].set_yticks([])
    ax[2].tick_params(labelsize=other_font)
    ax[2].set_xlabel('Taps', fontsize=other_font)

    ax[3].set_title('Frequency response of filter', fontsize=title_font)
    ax[3].set_xlim([0, max_frequency])
    ax[3].legend(loc='upper right', fontsize=other_font)
    ax[3].tick_params(labelsize=other_font)
    ax[3].set_xlabel('Frequency [Hz]', fontsize=other_font)

    plt.tight_layout()
    plt.show()
