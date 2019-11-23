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

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.nn.spectrum import compute_wavelets, apply_wavelets_rectangular
from sleeprnn.common import pkeys, constants, viz
from sleeprnn.data.utils import power_spectrum
from sleeprnn.helpers import plotter

CUSTOM_COLORS = {'red': '#c62828', 'grey': '#455a64', 'blue': '#0277bd', 'green': '#43a047'}


def filter_stamps(stamps, start_sample, end_sample):
    pages_list = []
    for i in range(stamps.shape[0]):
        start_inside = (stamps[i, 0] > start_sample) and (stamps[i, 0] < end_sample)
        end_inside = (stamps[i, 1] > start_sample) and (stamps[i, 1] < end_sample)

        if start_inside or end_inside:
            pages_list.append(stamps[i, :])
    return pages_list


if __name__ == '__main__':
    database_name = constants.MASS_SS_NAME
    subject_id = 1
    location = 100  # 337  # 425
    location_is_time = False  # If False, is stamp idx
    show_magnitude = False

    init_fb = 0.5
    border_duration = 5
    context_duration = 15
    n_scales = 6
    upper_freq = 13
    lower_freq = 1

    if database_name == constants.MASS_SS_NAME:
        dataset = load_dataset(
            constants.MASS_KC_NAME,
            params={pkeys.NORM_COMPUTATION_MODE: constants.NORM_GLOBAL})
        y_kc = dataset.get_subject_stamps(subject_id=subject_id)
    else:
        y_kc = None

    dataset = load_dataset(
        database_name,
        params={pkeys.NORM_COMPUTATION_MODE: constants.NORM_GLOBAL})
    fs = dataset.fs
    x = dataset.get_subject_signal(subject_id=subject_id, normalize_clip=False)
    y = dataset.get_subject_stamps(subject_id=subject_id)

    border_size = int(fs * border_duration)
    context_size = int((context_duration + 2*border_duration) * fs)

    if location_is_time:
        center_sample = int(location * fs)
    else:
        center_sample = int(y[location, :].mean())

    start_sample = center_sample - (context_size // 2)
    end_sample = center_sample + (context_size // 2)
    demo_signal = x[start_sample:end_sample]

    print('Length of input:', demo_signal.size)
    start_sample_display = start_sample + border_size
    end_sample_display = end_sample - border_size
    time_axis = np.arange(end_sample_display - start_sample_display) / fs

    # Eligible stamps
    segment_stamps = filter_stamps(y, start_sample_display, end_sample_display)
    if y_kc is not None:
        segment_stamps_kc = filter_stamps(y_kc, start_sample_display, end_sample_display)
    else:
        segment_stamps_kc = None

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
    #x_ticks_major = np.ceil(time_axis[0]) + np.arange(context_duration)
    x_ticks_major = []
    x_ticks_minor = np.arange(context_duration)
    fig = plt.figure(figsize=(8, 6), dpi=200)
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 2, 6, 1])

    # KERNEL
    ax = fig.add_subplot(gs[0])
    kernel_size_display = int(2 * fs) + 1
    wavelet_to_show_idx = n_scales // 2
    wavelet_to_show_real = kernels_real[:, wavelet_to_show_idx]
    wavelet_to_show_imag = kernels_imag[:, wavelet_to_show_idx]
    ntaps = wavelet_to_show_real.size

    max_kernel_size_tmp = 5001
    kernel_base = np.zeros(max_kernel_size_tmp)
    half_sample = kernel_base.size // 2
    kernel_full_real = kernel_base.copy()
    kernel_full_imag = kernel_base.copy()
    kernel_half_size = ntaps // 2
    start_kernel = half_sample - kernel_half_size
    end_kernel = half_sample + kernel_half_size + 1
    kernel_full_real[start_kernel:end_kernel] = wavelet_to_show_real
    kernel_full_imag[start_kernel:end_kernel] = wavelet_to_show_imag
    start_crop = int(half_sample - kernel_size_display//2)
    end_crop = start_crop + kernel_size_display
    kernel_crop_real = kernel_full_real[start_crop:end_crop]
    kernel_crop_imag = kernel_full_imag[start_crop:end_crop]
    max_value_kernel = np.max(np.abs(kernel_crop_real))
    kernel_x_axis = np.arange(kernel_crop_real.size) - kernel_crop_real.size // 2

    ax.plot(
        kernel_x_axis, kernel_crop_imag,
        linewidth=1, color=CUSTOM_COLORS['red'], label='Imaginary Part')
    ax.plot(
        kernel_x_axis, kernel_crop_real,
        linewidth=1, color=CUSTOM_COLORS['blue'], label='Real Part')
    ax.set_yticks([])
    ax.set_title(
        'Mother Wavelet with $F_B$=%1.1f' % init_fb, loc='left',
        fontsize=viz.FONTSIZE_TITLE)
    ax.set_xticks([])
    ax.set_xlim([kernel_x_axis[0], kernel_x_axis[-1]])
    ax.set_ylim([-1.1*max_value_kernel, 1.1*max_value_kernel])
    ax.legend(
        loc='upper right', fontsize=viz.FONTSIZE_GENERAL, ncol=2,
        bbox_to_anchor=(1, 1.4), frameon=False
    )
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax = plotter.set_axis_color(ax)

    # ORIGINAL SIGNAL
    ax = fig.add_subplot(gs[1])
    y_max = 150
    ax.plot(
        time_axis, demo_signal.flatten()[border_size:-border_size],
        linewidth=0.8, color=CUSTOM_COLORS['grey'])
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    stamp_label_used = False
    for expert_stamp in segment_stamps:
        if stamp_label_used:
            label = None
        else:
            label = 'Sleep Spindle'
            stamp_label_used = True
        ax.fill_between(
            (expert_stamp - start_sample_display) / fs, y_max, -y_max,
            facecolor=CUSTOM_COLORS['blue'], alpha=0.3, label=label)
    stamp_label_used = False
    if segment_stamps_kc is not None:
        for expert_stamp in segment_stamps_kc:
            if stamp_label_used:
                label = None
            else:
                label = 'K-Complex'
                stamp_label_used = True
            ax.fill_between(
                (expert_stamp - start_sample_display) / fs, y_max, -y_max,
                facecolor=CUSTOM_COLORS['green'], alpha=0.3, label=label)

    ax.set_ylim([-y_max, y_max])
    ax.set_yticks([-50, 50])
    ax.set_yticklabels(['-50 uV', '50 uV'], fontsize=viz.FONTSIZE_GENERAL)
    ax.set_xlim([time_axis[0], time_axis[-1]])
    ax.legend(
        loc='upper right', fontsize=viz.FONTSIZE_GENERAL, ncol=2,
        bbox_to_anchor=(1, 1.5), frameon=False
    )
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel('Intervals of 1 [s]', fontsize=viz.FONTSIZE_GENERAL)
    ax.set_title(
        'Input Signal', loc='left',
        fontsize=viz.FONTSIZE_TITLE)
    ax.set_xticks(x_ticks_major)
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.grid(b=True, axis='x', which='minor')
    ax = plotter.set_axis_color(ax)

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
            label_imag = 'Imaginary Part'
            label_magnitude = 'Magnitude'
        else:
            label_real = None
            label_imag = None
            label_magnitude = None
        center_scale = -i*scale_sep

        signal_to_show = results_imag[:, i] * compensation_factor_list[i]
        signal_to_show = (signal_to_show + max_value) / (2 * max_value) - 0.5
        signal_to_show_imag = signal_to_show + center_scale

        signal_to_show = results_real[:, i] * compensation_factor_list[i]
        signal_to_show = (signal_to_show + max_value) / (2 * max_value) - 0.5
        signal_to_show_real = signal_to_show + center_scale

        if show_magnitude:
            magnitude_to_show = np.sqrt(
                results_imag[:, i] ** 2 + results_real[:, i] ** 2
            ) * compensation_factor_list[i]
            magnitude_to_show = (magnitude_to_show + max_value) / (2 * max_value) - 0.5
            magnitude_to_show = magnitude_to_show + center_scale
            ax.fill_between(
                time_axis, magnitude_to_show, center_scale,
                linewidth=1, color=CUSTOM_COLORS['blue'], label=label_magnitude)
        else:
            ax.plot(
                time_axis, signal_to_show_imag,
                linewidth=1, color=CUSTOM_COLORS['red'], label=label_imag)
            ax.plot(
                time_axis, signal_to_show_real,
                linewidth=1, color=CUSTOM_COLORS['blue'], label=label_real)

    ax.set_yticks([-scale_sep * k for k in range(n_scales)])
    ax.set_yticklabels(
        ['%1.1f Hz' % freq for freq in frequencies],
        fontsize=viz.FONTSIZE_GENERAL
    )
    #ax.set_yticklabels(np.round(frequencies, decimals=1))
    ax.set_xlim([time_axis[0], time_axis[-1]])
    # ax.set_ylabel('Central Frequency [Hz]', fontsize=7)
    ax.set_title(
        'Continuous Wavelet Transform (CWT)', loc='left',
        fontsize=viz.FONTSIZE_TITLE)
    ax.set_xlabel('Intervals of 1 [s]', fontsize=viz.FONTSIZE_GENERAL)
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.legend(
        loc='upper right', fontsize=viz.FONTSIZE_GENERAL, ncol=2,
        bbox_to_anchor=(1, 1.15), frameon=False
    )
    ax.set_xticks(x_ticks_major)
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.grid(b=True, axis='x', which='minor')
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax = plotter.set_axis_color(ax)

    # Response
    ax = fig.add_subplot(gs[3])
    max_value_fft = 0
    for i in range(n_scales):
        freq_axis = freq_axis_list[i]
        fft_kernel = fft_kernel_list[i]
        ax.plot(freq_axis, fft_kernel, linewidth=1, color=viz.PALETTE['grey'])
        if np.max(fft_kernel) > max_value_fft:
            max_value_fft = np.max(fft_kernel)
    ax.set_xscale('log')
    ax.set_yticks([])
    ax.set_xlim([freq_axis[0], freq_axis[-1]])
    # ax.set_xlabel('Frequency [Hz]', fontsize=viz.FONTSIZE_GENERAL)
    ax.set_title(
        "Frequency Response of Wavelets", loc='left',
        fontsize=viz.FONTSIZE_TITLE)
    ax.set_ylim([0, 1.1*max_value_fft])
    ax.tick_params(labelsize=viz.FONTSIZE_GENERAL)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks([1, 10])
    ax.set_xticklabels(['1 Hz', '10 Hz'], fontsize=viz.FONTSIZE_GENERAL)
    ax = plotter.set_axis_color(ax)

    plt.tight_layout()
    plt.show()

    pool_size = n_scales // 8
    for i in range(8):
        subset_freq = frequencies[i*pool_size:(i+1)*pool_size]
        print(subset_freq)
