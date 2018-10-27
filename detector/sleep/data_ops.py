"""data_ops.py: Module for general sleep eeg data operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
from scipy.signal import resample_poly, butter, filtfilt

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_DATA = os.path.abspath(os.path.join(PATH_THIS_DIR, '../../data'))


def seq2inter(sequence):
    """Returns the start and end samples of active intervals in a binary sequence."""
    if not np.array_equal(sequence, sequence.astype(bool)):
        raise Exception('Sequence must have binary values only')
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
    """Returns the binary sequence segment from 'start' to 'end', associated with the active intervals."""
    if np.sum((intervals < start)) > 0 or np.sum((intervals > end)) > 0:
        raise Exception('Values in inter matrix should be within start and end bounds')
    sequence = np.zeros(end - start + 1, dtype=np.int32)
    for i in range(len(intervals)):
        start_sample = intervals[i, 0] - start
        end_sample = intervals[i, 1] - start + 1
        sequence[start_sample:end_sample] = 1
    return sequence

# TODO: test filter
def filter_eeg(signal, fs, lowcut=0.5, highcut=35):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz bandpass."""
    # Generate butter bandpass of order 3
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(3, [low, high], btype='band')
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

# TODO: implement norm and clip using median and interquartile
def norm_clip_eeg(signal, n2_pages, page_size, clip_value=3):
    """Normalizes EEG data according to N2 pages statistics, and then clips extreme values.

    EEGs are very close to a Gaussian signal, but are subject to outlier values. To compute a more robust
    estimation of the underlying mean and variance of N2 pages, we compute the median and the interquartile
    range. These estimations are used to normalize the signal with a Z-score. After normalization, the signal
    is clipped to the [-clip_value, clip_value] range.

    Args:
        signal: 1-D array containing EEG data.
        n2_pages: 1-D array with indices of N2 pages of the hypnogram.
        page_size: (int) Number of samples contained in a single page of the hypnogram.
        clip_value: (Optional, Defaults to 3) Value used to clip the signal after normalization.
    """
    return signal


def combine_close_marks(marks, fs, min_separation):
    """Combines contiguous marks that are too close to each other."""
    marks = np.sort(marks, axis=0)
    combined_marks = [marks[0, :]]
    for i in range(1, marks.shape[0]):
        last_mark = combined_marks[-1]
        this_mark = marks[i, :]
        gap = (this_mark[0] - last_mark[1]) / fs
        if gap < min_separation:
            # Combine mark, so the last mark ends where this mark ends.
            combined_marks[-1][1] = this_mark[1]
        else:
            combined_marks.append(this_mark)
    combined_marks = np.stack(combined_marks, axis=0)
    return combined_marks


def filter_duration_marks(marks, fs, min_duration, max_duration):
    """Removes marks that are too short or too long. Marks are assumed to be sample-stamps."""
    durations = (marks[:, 1] - marks[:, 0]) / fs
    feasible_idx = np.where((durations >= min_duration) & (durations <= max_duration))[0]
    filtered_marks = marks[feasible_idx, :]
    return filtered_marks
