from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf

detector_path = os.path.abspath('..')
print(detector_path)
sys.path.append(detector_path)

from sleeprnn.data.loader import load_dataset
from sleeprnn.nn.spectrum import compute_wavelets, apply_wavelets_rectangular
from sleeprnn.common import pkeys, constants
from sleeprnn.data.utils import power_spectrum

CUSTOM_COLORS = {'red': '#c62828', 'grey': '#455a64', 'blue': '#0277bd', 'green': '#43a047'}

if __name__ == '__main__':
    init_fb = 0.5
    border_duration = 5
    context_duration = 6
    n_scales = 8
    upper_freq = 20
    lower_freq = 1

    dataset = load_dataset(
        constants.MASS_SS_NAME,
        params={pkeys.NORM_COMPUTATION_MODE: constants.NORM_GLOBAL})
    fs = dataset.fs
    x = dataset.get_subject_signal(subject_id=2, normalize_clip=True)
    y = dataset.get_subject_stamps(subject_id=2)
    which_stamp = 426  # 337  # 425

    border_size = int(fs * border_duration)
    context_size = int((context_duration + 2*border_duration) * fs)
    center_sample = int(y[which_stamp, :].mean())
    start_sample = center_sample - (context_size // 2)
    end_sample = center_sample + (context_size // 2)
    demo_signal = x[start_sample:end_sample]

    print('Length of input:', demo_signal.size)
    time_axis = (np.arange(demo_signal.size) / fs)[border_size:-border_size]

    # Build computational
    tf.reset_default_graph()
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, demo_signal.size],
                            name="feats_train_ph")

    wavelets, frequencies = compute_wavelets(
        fb_list=[init_fb],
        fs=fs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        n_scales=n_scales,
        size_factor=1.0,
        flattening=False,
        trainable=False,
        name='cmorlet')

    outputs = apply_wavelets_rectangular(
        inputs=inputs,
        wavelets=wavelets,
        border_crop=border_size,
        stride=1,
        name='cwt')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    results, kernels = sess.run([outputs, wavelets], feed_dict={inputs: [demo_signal]})
    results_real = results[0, ..., 0]
    results_imag = results[0, ..., 1]
    kernels_real = np.squeeze(kernels[0][0])
    kernels_imag = np.squeeze(kernels[0][1])

    print(results_real.shape, results_imag.shape, kernels_real.shape, kernels_imag.shape)

    # Show results
    fig = plt.figure(figsize=(10, 8), dpi=100)
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 6, 1])

    # ORIGINAL SIGNAL
    ax = fig.add_subplot(gs[0])
    ax.plot(
        time_axis, demo_signal.flatten()[border_size:-border_size],
        label='Original', linewidth=1, color=CUSTOM_COLORS['grey'])
    # ax.set_ylim([-5, 5])
    ax.set_yticks([])
    ax.set_xlim([time_axis[0], time_axis[-1]])
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(labelsize=8)

    # KERNEL
    ax = fig.add_subplot(gs[1])
    kernel_size_display = int(2 * fs) + 1
    wavelet_to_show_idx = n_scales // 2
    wavelet_to_show = kernels_real[:, wavelet_to_show_idx]
    ntaps = wavelet_to_show.size

    max_kernel_size_tmp = 5001
    kernel_base = np.zeros(max_kernel_size_tmp)
    half_sample = kernel_base.size // 2
    kernel_full = kernel_base.copy()
    kernel_half_size = wavelet_to_show.size // 2
    start_kernel = half_sample - kernel_half_size
    end_kernel = half_sample + kernel_half_size + 1
    kernel_full[start_kernel:end_kernel] = wavelet_to_show
    start_crop = int(half_sample - kernel_size_display//2)
    end_crop = start_crop + kernel_size_display
    kernel_crop = kernel_full[start_crop:end_crop]
    max_value_kernel = np.max(np.abs(kernel_crop))
    kernel_x_axis = np.arange(kernel_crop.size) - kernel_crop.size // 2

    ax.plot(
        kernel_x_axis, kernel_crop,
        linewidth=1, color=CUSTOM_COLORS['grey'],
        label='Max ntaps: %d (Fb %1.2f)' % (ntaps, init_fb))
    ax.set_yticks([])
    ax.set_xlim([kernel_x_axis[0], kernel_x_axis[-1]])
    ax.set_ylim([-max_value_kernel, max_value_kernel])
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(labelsize=8)

    # FREQ RESPONSE
    max_freq = 50
    min_freq = 0.1
    compensation_factor_list = []
    freq_axis_list = []
    fft_kernel_list = []
    for i in range(n_scales):
        this_kernel = kernels_real[:, i]
        max_kernel_size_to_fft = ntaps + 2000
        kernel_base = np.zeros(max_kernel_size_to_fft)
        half_sample = kernel_base.size // 2
        kernel_full = kernel_base.copy()
        kernel_half_size = this_kernel.size // 2
        start_kernel = half_sample - kernel_half_size
        end_kernel = half_sample + kernel_half_size + 1
        kernel_full[start_kernel:end_kernel] = this_kernel
        fft_kernel, freq_axis = power_spectrum(kernel_full, fs)
        compensation_factor = 1 / fft_kernel.max()
        compensation_factor_list.append(compensation_factor)
        fft_kernel = fft_kernel * compensation_factor
        fft_kernel = fft_kernel[freq_axis <= max_freq]
        freq_axis = freq_axis[freq_axis <= max_freq]
        fft_kernel = fft_kernel[freq_axis >= min_freq]
        freq_axis = freq_axis[freq_axis >= min_freq]
        freq_axis_list.append(freq_axis)
        fft_kernel_list.append(fft_kernel)

    # CWT RESULT
    ax = fig.add_subplot(gs[2])
    scale_sep = 1
    max_value = 0
    for i in range(n_scales):
        this_max = np.max(np.abs(results_real[:, i] * compensation_factor_list[i]))
        if this_max > max_value:
            max_value = this_max
    for i in range(n_scales):
        if i == 0:
            label_real = 'Real Part'
            label_imag = 'Imag Part'
        else:
            label_real = None
            label_imag = None
        center_scale = -i*scale_sep

        signal_to_show = results_imag[:, i] * compensation_factor_list[i]
        signal_to_show = (signal_to_show + max_value) / (2 * max_value) - 0.5
        signal_to_show = signal_to_show + center_scale
        ax.plot(
            time_axis, signal_to_show,
            linewidth=1, color=CUSTOM_COLORS['red'], label=label_imag)

        signal_to_show = results_real[:, i] * compensation_factor_list[i]
        signal_to_show = (signal_to_show + max_value) / (2 * max_value) - 0.5
        signal_to_show = signal_to_show + center_scale
        ax.plot(
            time_axis, signal_to_show,
            linewidth=1, color=CUSTOM_COLORS['blue'], label=label_real)

    ax.set_yticks([-scale_sep * k for k in range(n_scales)])
    ax.set_yticklabels(np.round(frequencies, decimals=1))
    ax.set_xlim([time_axis[0], time_axis[-1]])
    ax.set_ylabel('Central Frequency [Hz]', fontsize=8)
    ax.set_xlabel('Time [s]', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(loc='upper right', fontsize=8)

    # Response
    ax = fig.add_subplot(gs[3])
    for i in range(n_scales):
        freq_axis = freq_axis_list[i]
        fft_kernel = fft_kernel_list[i]
        ax.plot(freq_axis, fft_kernel)
    ax.set_xscale('log')
    ax.set_yticks([])
    ax.set_xlim([freq_axis[0], freq_axis[-1]])
    ax.set_xlabel('Frequency [Hz]', fontsize=8)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()
