"""postprocessing.py: Module for general postprocessing operations of predicted annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def combine_close_marks(marks, fs, min_separation):
    """Combines contiguous marks that are too close to each other. Marks are assumed to be sample-stamps."""
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
