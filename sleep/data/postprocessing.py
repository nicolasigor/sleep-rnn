"""postprocessing.py: Module for general postprocessing operations of
annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .data_ops import seq2inter_with_pages


def combine_close_marks(marks, fs, min_separation):
    """Combines contiguous marks that are too close to each other. Marks are
    assumed to be sample-stamps.

    If min_separation is None, the functionality is bypassed.
    """
    if min_separation is None:
        combined_marks = marks
    else:
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
    """Removes marks that are too short or strangely long. Marks longer than
    max_duration but not strangely long are cropped to keep the central
    max_duration duration. Durations are assumed to be in seconds.
    Marks are assumed to be sample-stamps.

    If min_duration is None, no short marks are removed.
    If max_duration is None, no long marks are removed.
    """
    durations = (marks[:, 1] - marks[:, 0]) / fs

    if min_duration is not None:
        # Remove too short spindles
        feasible_idx = np.where(durations >= min_duration)[0]
        marks = marks[feasible_idx, :]
        durations = durations[feasible_idx]

    if max_duration is not None:
        # Remove weird annotations (extremely long)
        feasible_idx = np.where(durations <= 2*max_duration)[0]
        marks = marks[feasible_idx, :]
        durations = durations[feasible_idx]
        # For annotations with durations longer than max_duration,
        # keep the central seconds
        excess = durations - max_duration
        excess = np.clip(excess, 0, None)
        half_remove = (fs * excess / 2).astype(np.int32)
        marks[:, 0] = marks[:, 0] + half_remove
        marks[:, 1] = marks[:, 1] - half_remove

    return marks


def upsample_mark_intervals(marks, upsample_factor=8):
    """Upsamples timestamps of marks to match a greater sampling frequency.
    """
    marks = marks * upsample_factor
    marks[:, 0] = marks[:, 0] - upsample_factor / 2
    marks[:, 1] = marks[:, 1] + upsample_factor / 2
    marks = marks.astype(np.int32)
    return marks


def generate_mark_intervals(
        pages_sequence,
        pages_indices,
        fs_input,
        fs_output,
        thr=0.5,
        min_separation=None,
        min_duration=None,
        max_duration=None,
):
    """
    If thr is None, pages_sequence is assumed to be already binarized.
    fs_input corresponds to sampling frequency of pages_sequence,
    fs_outputs corresponds to desired sampling frequency.
    """
    # Thresholding
    if thr is not None:
        pages_sequence_bin = (pages_sequence >= thr).astype(np.int32)
    else:
        pages_sequence_bin = pages_sequence

    # Transformation to intervals
    mark_intervals = seq2inter_with_pages(pages_sequence_bin, pages_indices)

    # Postprocessing
    # Note that when min_separation, min_duration, or max_duration is None,
    # that postprocessing doesn't happen.
    mark_intervals = combine_close_marks(
        mark_intervals, fs_input, min_separation)
    mark_intervals = filter_duration_marks(
        mark_intervals, fs_input, min_duration, max_duration)

    # Upsampling
    if fs_output > fs_input:
        mark_intervals = upsample_mark_intervals(
            mark_intervals, upsample_factor=fs_output/fs_input)
    elif fs_output < fs_input:
        raise ValueError('fs_output has to be greater than fs_input')

    return mark_intervals


def generate_mark_intervals_with_list(
        pages_sequence_list,
        pages_indices_list,
        fs_input,
        fs_output,
        thr=0.5,
        min_separation=None,
        min_duration=None,
        max_duration=None,
):
    mark_intervals_list = [
        generate_mark_intervals(
            pages_sequence,
            pages_indices,
            fs_input,
            fs_output,
            thr,
            min_separation,
            min_duration,
            max_duration)
        for (pages_sequence, pages_indices)
        in zip(pages_sequence_list, pages_indices_list)
    ]
    return mark_intervals_list

