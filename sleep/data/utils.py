"""utils.py: Module for general data eeg data operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly, butter, filtfilt, firwin, lfilter
from scipy.stats import iqr

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_DATA = os.path.join(PATH_THIS_DIR, '..', '..', 'resources', 'datasets')

from sleep.common import constants, checks
from .inta_ss import IntaSS
from .mass_kc import MassKC
from .mass_ss import MassSS


def load_dataset(dataset_name, load_checkpoint=True, params=None):
    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [
            constants.MASS_KC_NAME,
            constants.MASS_SS_NAME,
            constants.INTA_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=load_checkpoint, params=params)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(load_checkpoint=load_checkpoint, params=params)
    else:
        dataset = IntaSS(load_checkpoint=load_checkpoint, params=params)
    return dataset


def seq2stamp(sequence):
    """Returns the start and end samples of stamps in a binary sequence."""
    if not np.array_equal(sequence, sequence.astype(bool)):
        raise ValueError('Sequence must have binary values only')
    n = len(sequence)
    tmp_result = np.diff(sequence, prepend=0)
    start_times = np.where(tmp_result == 1)[0]
    end_times = np.where(tmp_result == -1)[0] - 1
    # Final edge case
    if start_times.size > end_times.size:
        end_times = np.concatenate([end_times, [n - 1]])
    stamps = np.stack([start_times, end_times], axis=1)
    return stamps


def stamp2seq(stamps, start, end, allow_early_end=False):
    """Returns the binary sequence segment from 'start' to 'end',
    associated with the stamps."""
    if np.any(stamps < start):
        msg = 'Values in intervals should be within start bound'
        raise ValueError(msg)
    if np.any(stamps > end) and not allow_early_end:
        msg = 'Values in intervals should be within end bound'
        raise ValueError(msg)

    sequence = np.zeros(end - start + 1, dtype=np.int32)
    for i in range(len(stamps)):
        start_sample = stamps[i, 0] - start
        end_sample = stamps[i, 1] - start + 1
        sequence[start_sample:end_sample] = 1
    return sequence


def seq2stamp_with_pages(pages_sequence, pages_indices):
    """Returns the start and end samples of stamps in a binary sequence that
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
    stamps = seq2stamp(global_sequence)
    return stamps


def broad_filter(signal, fs, lowcut=0.1, highcut=35):
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


def narrow_filter(signal, fs, lowcut, highcut):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz
    bandpass."""
    ntaps = 21
    width = 0.5
    cutoff = [lowcut, highcut]
    b = firwin(ntaps, cutoff, width, pass_zero=False, fs=fs)
    filtered_signal = lfilter(b, [1.0], signal)
    filtered_signal = np.append(np.array([0, 0]), filtered_signal) - np.append(
        filtered_signal, np.array([0, 0]))
    filtered_signal = filtered_signal[0:len(filtered_signal) - 2]
    return filtered_signal


def resample_signal(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz."""
    gcd_freqs = math.gcd(fs_new, fs_old)
    up = int(fs_new / gcd_freqs)
    down = int(fs_old / gcd_freqs)
    signal = resample_poly(signal, up, down)
    signal = np.array(signal, dtype=np.float32)
    return signal


def resample_signal_linear(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz.

    This implementation uses simple linear interpolation to achieve this.
    """
    t = np.cumsum(np.ones(len(signal)) / fs_old)
    t_new = np.arange(t[0], t[-1], 1 / fs_new)
    signal = interp1d(t, signal)(t_new)
    return signal


def norm_clip_signal(signal, pages_indices, page_size, clip_value=10):
    """Normalizes EEG data according to N2 pages statistics, and then clips.

    EEGs are very close to a Gaussian signal, but are subject to outlier values.
    To compute a more robust estimation of the underlying mean and variance of
    pages, we compute the median and the interquartile range. These
    estimations are used to normalize the signal with a Z-score. After
    normalization, the signal is clipped to the [-clip_value, clip_value] range.

    Args:
        signal: 1-D array containing EEG data.
        pages_indices: 1-D array with indices of pages of the hypnogram.
        page_size: (int) Number of samples contained in a single page of the
            hypnogram.
        clip_value: (Optional, int, Defaults to 6) Value used to clip the signal
            after normalization.
    """
    # Extract statistics only from N2 stages
    n2_data = extract_pages(signal, pages_indices, page_size)
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
    pages_data = np.stack(pages_list, axis=0)
    return pages_data


def extract_pages_for_stamps(stamps, pages_indices, page_size):
    """Returns stamps that are at least partially contained on pages."""
    pages_list = []
    for i in range(stamps.shape[0]):
        stamp_start_page = stamps[i, 0] // page_size
        stamp_end_page = stamps[i, 1] // page_size

        start_inside = (stamp_start_page in pages_indices)
        end_inside = (stamp_end_page in pages_indices)

        if start_inside or end_inside:
            pages_list.append(stamps[i, :])

    pages_data = np.stack(pages_list, axis=0)
    return pages_data


def simple_split_with_list(x, y, train_fraction=0.8, seed=None):
    """Splits data stored in a list.

    The data x and y are list of arrays with shape [batch, ...].
    These are split in two sets randomly using train_fraction over the number of
    element of the list. Then these sets are returned with
    the arrays concatenated along the first dimension
    """
    n_subjects = len(x)
    n_train = int(n_subjects * train_fraction)
    print('Split: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    x_train = np.concatenate([x[i] for i in train_idx], axis=0)
    y_train = np.concatenate([y[i] for i in train_idx], axis=0)
    x_test = np.concatenate([x[i] for i in test_idx], axis=0)
    y_test = np.concatenate([y[i] for i in test_idx], axis=0)
    return x_train, y_train, x_test, y_test


def split_ids_list(subject_ids, train_fraction=0.75, seed=None, verbose=True):
    """Splits the subject_ids list randomly using train_fraction."""
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_fraction)
    if verbose:
        print('Split IDs: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    train_ids = [subject_ids[i] for i in train_idx]
    test_ids = [subject_ids[i] for i in test_idx]
    return train_ids, test_ids


def shuffle_data(x, y, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = x.shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    x = x[random_idx]
    y = y[random_idx]
    return x, y
