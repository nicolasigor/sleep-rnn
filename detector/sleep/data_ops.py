"""data_ops.py: Module for general sleep eeg data operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
from scipy.stats import iqr

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_DATA = os.path.join(PATH_THIS_DIR, '..', '..', 'data')


def seq2inter(sequence):
    """Returns the start and end samples of intervals in a binary sequence."""
    if not np.array_equal(sequence, sequence.astype(bool)):
        raise ValueError('Sequence must have binary values only')
    intervals = []
    n = len(sequence)
    prev_val = 0
    for i in range(n):
        if sequence[i] > prev_val:      # We just turned on
            intervals.append([i, i])
        elif sequence[i] < prev_val:    # We just turned off
            intervals[-1][1] = i-1
        prev_val = sequence[i]
    if sequence[-1] == 1:
        intervals[-1][1] = n-1
    intervals = np.stack(intervals)
    return intervals


def inter2seq(intervals, start, end):
    """Returns the binary sequence segment from 'start' to 'end',
    associated with the active intervals."""
    if np.any(intervals < start) or np.any(intervals > end):
        msg = 'Values in intervals should be within start and end bounds'
        raise ValueError(msg)
    sequence = np.zeros(end - start + 1, dtype=np.int32)
    for i in range(len(intervals)):
        start_sample = intervals[i, 0] - start
        end_sample = intervals[i, 1] - start + 1
        sequence[start_sample:end_sample] = 1
    return sequence


def filter_eeg(signal, fs, lowcut=0.5, highcut=35):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz
    bandpass."""
    # Generate butter bandpass of order 3.
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(3, [low, high], btype='band')
    # Apply filter to the signal with zero-phase.
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def resample_eeg(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz."""
    gcd_freqs = math.gcd(fs_new, fs_old)
    up = int(fs_new / gcd_freqs)
    down = int(fs_old / gcd_freqs)
    signal = resample_poly(signal, up, down)
    signal = np.array(signal, dtype=np.float32)
    return signal


def norm_clip_eeg(signal, n2_pages_indices, page_size, clip_value=6):
    """Normalizes EEG data according to N2 pages statistics, and then clips.

    EEGs are very close to a Gaussian signal, but are subject to outlier values.
    To compute a more robust estimation of the underlying mean and variance of
    N2 pages, we compute the median and the interquartile range. These
    estimations are used to normalize the signal with a Z-score. After
    normalization, the signal is clipped to the [-clip_value, clip_value] range.

    Args:
        signal: 1-D array containing EEG data.
        n2_pages_indices: 1-D array with indices of N2 pages of the hypnogram.
        page_size: (int) Number of samples contained in a single page of the
            hypnogram.
        clip_value: (Optional, int, Defaults to 6) Value used to clip the signal
            after normalization.
    """
    # Extract statistics only from N2 stages
    n2_data = extract_pages(signal, n2_pages_indices, page_size)
    n2_signal = np.concatenate(n2_data)
    signal_std = iqr(n2_signal) / 1.349
    signal_median = np.median(n2_signal)

    # Normalize entire signal
    norm_signal = (signal - signal_median) / signal_std

    # Now clip to clip_value (only if clip is not None)
    if clip_value:
        norm_signal = np.clip(norm_signal, -clip_value, clip_value)
    return norm_signal


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


def extract_pages(sequence, pages_indices, page_size, border_size=0):
    """Extracts and returns the given set of pages from the sequence.

    Args:
        sequence: (1-D Array) sequence from where to extract data.
        pages_indices: (1-D Array) array of indices of pages to be extracted.
        page_size: (int) number in samples of each page.
        border_size: (Optional, int,, defaults to 0) number of samples to be
            added at each border.

    Returns:
        pages_data: (2-D Array) array of shape [n_pages,page_size+2*border_size]
            that contains the extracted data.
    """
    pages_list = []
    for page in pages_indices:
        sample_start = page * page_size - border_size
        sample_end = (page + 1) * page_size + border_size
        page_signal = sequence[sample_start:sample_end]
        pages_list.append(page_signal)
    pages_data = np.stack(pages_list)
    return pages_data


def seq2inter_with_pages(pages_sequence, pages_indices):
    """Returns the start and end samples of intervals in a binary sequence that
    is split in pages."

    Args:
        pages_sequence: (2d array) binary array with shape [n_pages, page_size]
        pages_indices: (1d array) array of indices of the corresponding pages in
            pages_sequence, with shape [n_pages,]
    """
    if pages_sequence.shape[0] != pages_indices.shape[0]:
        raise ValueError('Shape mismatch. Inputs need the same number of rows.')
    tmp_sequence = pages_sequence.flatten()
    if not np.array_equal(tmp_sequence, tmp_sequence.astype(bool)):
        raise ValueError('Sequence must have binary values only')

    page_size = pages_sequence.shape[1]
    max_page = np.max(pages_indices)
    max_size = (max_page + 1) * page_size
    global_sequence = np.zeros(max_size, dtype=np.int32)
    for i, page in enumerate(pages_indices):
        sample_start = page * page_size
        sample_end = (page + 1) * page_size
        global_sequence[sample_start:sample_end] = pages_sequence[i, :]
    inter = seq2inter(global_sequence)
    return inter
