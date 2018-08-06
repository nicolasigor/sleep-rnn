from __future__ import division
from __future__ import print_function
import numpy as np


def seq2inter(sequence):
    if not np.array_equal(sequence, sequence.astype(bool)):
        raise Exception('Sequence must have binary values only')
    interval = []
    n = len(sequence)
    prev_val = 0
    for i in range(n):
        if sequence[i] > prev_val:      # We just turned on
            interval.append([i, i])
        elif sequence[i] < prev_val:    # We just turned off
            interval[-1][1] = i-1
        prev_val = sequence[i]
    if sequence[-1] == 1:
        interval[-1][1] = n-1
    interval = np.stack(interval)
    return interval


def inter2seq(inter, start, end):
    if (inter < start).sum() > 0 or (inter > end).sum() > 0:
        raise Exception('Values in inter matrix should be within start and end bounds')
    sequence = np.zeros(end - start + 1, dtype=np.int32)
    for i in range(len(inter)):
        start_sample = inter[i, 0] - start - 1
        end_sample = inter[i, 1] - start
        sequence[start_sample:end_sample] = 1
    return sequence
