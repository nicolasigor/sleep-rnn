from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import firwin
import tensorflow as tf


def apply_fir_filter_tf(signal, kernel):
    """For single signal, not batch"""
    signal = tf.reshape(signal, shape=[1, 1, -1, 1])
    kernel = tf.reshape(kernel, shape=[1, -1, 1, 1])
    with tf.device("/cpu:0"):
        new_signal = tf.nn.conv2d(
            input=signal, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    new_signal = new_signal[0, 0, :, 0]
    return new_signal


def random_window_tf(signal_size, window_min_size, window_max_size):
    window_size = tf.random.uniform([], minval=window_min_size, maxval=window_max_size)
    start_sample = tf.random.uniform([], minval=0, maxval=(signal_size - window_size - 1))
    k_array = np.arange(signal_size)
    offset_1 = start_sample + 0.1 * window_size
    offset_2 = start_sample + 0.9 * window_size
    scaling = 0.1 * window_size / 4
    window_onset = tf.math.sigmoid((k_array - offset_1) / scaling)
    window_offset = tf.math.sigmoid((k_array - offset_2) / scaling)
    window = window_onset - window_offset
    return window


def random_smooth_function_tf(signal_size, function_min_val, function_max_val, lp_filter_size):
    lp_filter = np.hanning(lp_filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    noise_vector = tf.random.uniform([signal_size], minval=-1, maxval=1)
    noise_vector = apply_fir_filter_tf(noise_vector, lp_filter)
    # Set noise to [0, 1] range
    min_val = tf.reduce_min(noise_vector)
    max_val = tf.reduce_max(noise_vector)
    noise_vector = (noise_vector - min_val) / (max_val - min_val)
    # Set to [function_min_val, function_max_val] range
    noise_vector = function_min_val + noise_vector * (function_max_val - function_min_val)
    return noise_vector


def lowpass_tf(signal, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hamming", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    new_signal = apply_fir_filter_tf(signal, lp_kernel)
    return new_signal


def highpass_tf(signal, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hamming", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    # HP = delta - LP
    hp_kernel = -lp_kernel
    hp_kernel[numtaps//2] += 1
    new_signal = apply_fir_filter_tf(signal, hp_kernel)
    return new_signal


def bandpass_tf(signal, fs, lowcut, highcut, filter_duration_ref=6, wave_expansion_factor=0.5):
    new_signal = signal
    if lowcut is not None:
        new_signal = highpass_tf(
            new_signal, fs, lowcut, filter_duration_ref, wave_expansion_factor)
    if highcut is not None:
        new_signal = lowpass_tf(
            new_signal, fs, highcut, filter_duration_ref, wave_expansion_factor)
    return new_signal


def generate_soft_mask_from_labels_tf(labels, fs, mask_lp_filter_duration=0.2, use_background=True):
    lp_filter_size = int(fs * mask_lp_filter_duration)
    labels = tf.cast(labels, tf.float32)
    # Enlarge labels
    expand_filter = np.ones(lp_filter_size).astype(np.float32)
    expanded_labels = apply_fir_filter_tf(labels, expand_filter)
    expanded_labels = tf.clip_by_value(expanded_labels, 0, 1)
    # Now filter
    lp_filter = np.hanning(lp_filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    smooth_labels = apply_fir_filter_tf(expanded_labels, lp_filter)
    if use_background:
        soft_mask = 1 - smooth_labels
    else:
        soft_mask = smooth_labels
    return soft_mask


def generate_wave_tf(
    signal_size,  # Number of samples
    fs,  # [Hz]
    max_amplitude,  # signal units
    min_frequency,  # [Hz]
    max_frequency,  # [Hz]
    frequency_deviation,  # [Hz]
    min_duration,  # [s]
    max_duration,  # [s]
    mask,  # [0, 1]
    frequency_lp_filter_duration=0.5,  # [s]
    amplitude_lp_filter_duration=0.5,  # [s]
):
    # This is ok to be numpy
    window_min_size = int(fs * min_duration)
    window_max_size = int(fs * max_duration)
    frequency_lp_filter_size = int(fs * frequency_lp_filter_duration)
    amplitude_lp_filter_size = int(fs * amplitude_lp_filter_duration)
    # Oscillation
    central_freq = tf.random.uniform([], minval=min_frequency, maxval=max_frequency)
    lower_freq = central_freq - frequency_deviation
    upper_freq = central_freq + frequency_deviation
    wave_freq = random_smooth_function_tf(signal_size, lower_freq, upper_freq, frequency_lp_filter_size)
    wave_phase = 2 * np.pi * tf.math.cumsum(wave_freq) / fs
    oscillation = tf.math.cos(wave_phase)
    # Amplitude
    amplitude_high = tf.random.uniform([], minval=0, maxval=max_amplitude)
    amplitude_low = tf.random.uniform([], minval=0, maxval=amplitude_high)
    amplitude = random_smooth_function_tf(signal_size, amplitude_low, amplitude_high, amplitude_lp_filter_size)
    # Window
    window = random_window_tf(signal_size, window_min_size, window_max_size)
    # Total wave
    generated_wave = window * amplitude * oscillation
    # Optional masking
    if mask is not None:
        generated_wave = generated_wave * mask
    return generated_wave


def generate_anti_wave_tf(
    signal,
    signal_size, # number of samples
    fs,  # [Hz]
    lowcut,  # [Hz]
    highcut,  # [Hz]
    min_duration,  # [s]
    max_duration,  # [s]
    max_attenuation,  # [0, 1]
    mask  # [0, 1]
):
    # This is ok to be numpy
    window_min_size = int(fs * min_duration)
    window_max_size = int(fs * max_duration)
    # Oscillation (opposite sign of band signal) and attenuation factor
    oscillation = -bandpass_tf(signal, fs, lowcut, highcut)
    attenuation_factor = tf.random.uniform([], minval=0, maxval=max_attenuation)
    # Window
    window = random_window_tf(signal_size, window_min_size, window_max_size)
    # Total wave
    generated_wave = window * attenuation_factor * oscillation
    # Optional masking
    if mask is not None:
        generated_wave = generated_wave * mask
    return generated_wave