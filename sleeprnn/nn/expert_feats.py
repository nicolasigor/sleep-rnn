from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import firwin
import tensorflow as tf

from sleeprnn.common import constants


def apply_fir_filter_tf_batch(signals, kernel):
    """For batch of signals"""
    signals = signals[:, tf.newaxis, :, tf.newaxis]
    kernel = tf.reshape(kernel, shape=[1, -1, 1, 1])
    new_signals = tf.nn.conv2d(
        input=signals, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    new_signals = new_signals[:, 0, :, 0]
    return new_signals


def lowpass_tf_batch(signals, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hamming", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    new_signals = apply_fir_filter_tf_batch(signals, lp_kernel)
    return new_signals


def highpass_tf_batch(signals, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hamming", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    # HP = delta - LP
    hp_kernel = -lp_kernel
    hp_kernel[numtaps // 2] += 1
    new_signals = apply_fir_filter_tf_batch(signals, hp_kernel)
    return new_signals


def bandpass_tf_batch(signals, fs, lowcut, highcut, filter_duration_ref=6, wave_expansion_factor=0.5):
    new_signals = signals
    if lowcut is not None:
        new_signals = highpass_tf_batch(
            new_signals, fs, lowcut, filter_duration_ref, wave_expansion_factor)
    if highcut is not None:
        new_signals = lowpass_tf_batch(
            new_signals, fs, highcut, filter_duration_ref, wave_expansion_factor)
    return new_signals


def moving_average_tf(signals, lp_filter_size):
    lp_filter = np.hanning(lp_filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    results = apply_fir_filter_tf_batch(signals, lp_filter)
    return results


def zscore_tf(signals, dispersion_mode=constants.DISPERSION_STD_ROBUST):
    mean_signals = tf.reduce_mean(signals, axis=1, keepdims=True)
    signals = signals - mean_signals
    if dispersion_mode == constants.DISPERSION_MADE:
        std_signals = tf.reduce_mean(tf.math.abs(signals), axis=1, keepdims=True)
    elif dispersion_mode == constants.DISPERSION_STD:
        std_signals = tf.math.sqrt(tf.reduce_mean(signals ** 2, axis=1, keepdims=True))
    elif dispersion_mode == constants.DISPERSION_STD_ROBUST:
        prc_range = [10, 90]  # valid percentile range to avoid extreme values
        signal_size = signals.get_shape().as_list()[1]
        loc_prc_start = int(signal_size * prc_range[0] / 100)
        loc_prc_end = int(signal_size * prc_range[1] / 100)
        sorted_values = tf.sort(signals, axis=1)
        sorted_values_valid = sorted_values[:, loc_prc_start:loc_prc_end]
        std_signals = tf.math.sqrt(tf.reduce_mean(sorted_values_valid ** 2, axis=1, keepdims=True))
    else:
        raise ValueError()
    signals = signals / std_signals
    return signals


def log10_tf(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def a7_layer_tf(
        signals,
        fs,
        window_duration,
        window_duration_absSigPow,
        sigma_lowcut=11,
        sigma_highcut=16,
        use_log_absSigPow=True,
        use_log_relSigPow=True,
        use_log_sigCov=True,
        use_zscore_relSigPow=True,
        use_zscore_sigCov=True,
        use_zscore_sigCorr=False,
        remove_delta_in_cov=False,
        dispersion_mode=constants.DISPERSION_STD_ROBUST
):
    with tf.variable_scope("a7_layer"):
        lp_filter_size = int(fs * window_duration)
        if window_duration_absSigPow is None:
            window_duration_absSigPow = window_duration
        lp_filter_size_absSigPow = int(fs * window_duration_absSigPow)
        print("Moving window: Using %1.2f s (%d samples)" % (window_duration, lp_filter_size))
        print("Moving window in absSigPow: Using %1.2f s (%d samples)" % (
            window_duration_absSigPow, lp_filter_size_absSigPow))
        print("Z-score: Using '%s'" % dispersion_mode)

        signal_sigma = bandpass_tf_batch(signals, fs, sigma_lowcut, sigma_highcut)
        signal_no_delta = bandpass_tf_batch(signals, fs, 4.5, None)

        # absolute sigma power
        signal_sigma_squared = signal_sigma ** 2
        abs_sig_pow_raw = moving_average_tf(signal_sigma_squared, lp_filter_size_absSigPow)
        if use_log_absSigPow:
            abs_sig_pow = log10_tf(abs_sig_pow_raw + 1e-4)
            print("absSigPow: Using log10.")
        else:
            abs_sig_pow = abs_sig_pow_raw

        # relative sigma power
        signal_sigma_squared = signal_sigma ** 2
        abs_sig_pow_raw = moving_average_tf(signal_sigma_squared, lp_filter_size)
        signal_no_delta_squared = signal_no_delta ** 2
        abs_no_delta_pow_raw = moving_average_tf(signal_no_delta_squared, lp_filter_size)
        rel_sig_pow_raw = abs_sig_pow_raw / (abs_no_delta_pow_raw + 1e-6)
        if use_log_relSigPow:
            rel_sig_pow = log10_tf(rel_sig_pow_raw + 1e-4)
            print("relSigPow: Using log10.")
        else:
            rel_sig_pow = rel_sig_pow_raw
        if use_zscore_relSigPow:
            rel_sig_pow = zscore_tf(rel_sig_pow, dispersion_mode)
            print("relSigPow: Using z-score.")

        # sigma covariance
        sigma_centered = signal_sigma - tf.reduce_mean(signal_sigma, axis=1)
        if remove_delta_in_cov:
            broad_centered = signal_no_delta - tf.reduce_mean(signal_no_delta, axis=1)
            print("Removing delta band in covariance.")
        else:
            broad_centered = signals - tf.reduce_mean(signals, axis=1)
        sig_cov_raw = moving_average_tf(sigma_centered * broad_centered, lp_filter_size)
        sig_cov = tf.nn.relu(sig_cov_raw)  # no negatives
        if use_log_sigCov:
            sig_cov = log10_tf(sig_cov + 1)  # Add 1
            print("sigCov: Using log10(x+1).")
        if use_zscore_sigCov:
            sig_cov = zscore_tf(sig_cov, dispersion_mode)
            print("sigCov: Using z-score.")

        # sigma correlation
        sig_var_raw = moving_average_tf(sigma_centered ** 2, lp_filter_size)
        broad_var_raw = moving_average_tf(broad_centered ** 2, lp_filter_size)
        sig_std_raw = tf.math.sqrt(sig_var_raw)
        broad_stf_raw = tf.math.sqrt(broad_var_raw)
        sig_corr = sig_cov_raw / (sig_std_raw * broad_stf_raw + 1e-4)
        if use_zscore_sigCorr:
            sig_corr = zscore_tf(sig_corr, dispersion_mode)
            print("sigCorr: Using z-score.")

        a7_parameters = tf.stack([abs_sig_pow, rel_sig_pow, sig_cov, sig_corr], axis=2)
    return a7_parameters
