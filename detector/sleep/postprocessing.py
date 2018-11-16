"""postprocessing.py: Module for general postprocessing operations of
annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def combine_close_marks(marks, fs, min_separation):
    """Combines contiguous marks that are too close to each other. Marks are
    assumed to be sample-stamps."""
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
    Marks are assumed to be sample-stamps."""
    durations = (marks[:, 1] - marks[:, 0]) / fs
    # Remove too short spindles
    feasible_idx = np.where(durations >= min_duration)[0]
    marks = marks[feasible_idx, :]
    durations = durations[feasible_idx]
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
