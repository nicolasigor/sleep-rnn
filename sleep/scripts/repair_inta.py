"""
Repairs INTA expert marks according to valid index.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pyedflib

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from sleep.data.data_ops import PATH_DATA
from sleep.data.inta_ss import NAMES, PATH_INTA_RELATIVE, PATH_MARKS, PATH_REC
from sleep.data import data_ops


if __name__ == '__main__':

    channel = 0  # f4-c4 channel
    filename_format = 'NewFixedSS_%s.txt'
    inta_folder = os.path.join(PATH_DATA, PATH_INTA_RELATIVE)

    for name in NAMES:
        print('Fixing %s' % name)
        path_marks_file = os.path.abspath(os.path.join(
            inta_folder, PATH_MARKS, 'SS_%s.txt' % name))
        path_eeg_file = os.path.abspath(os.path.join(
            inta_folder, PATH_REC, '%s.rec' % name))

        # Read marks
        print('Loading %s' % path_marks_file)
        data = np.loadtxt(path_marks_file)
        for_this_channel = data[:, -1] == channel + 1
        data = data[for_this_channel]
        data = np.round(data).astype(np.int32)

        # Remove zero duration marks, and ensure that start time < end time
        new_data = []
        for i in range(data.shape[0]):
            if data[i, 0] > data[i, 1]:
                aux = data[i, 0]
                data[i, 0] = data[i, 1]
                data[i, 1] = aux
                new_data.append(data[i, :])
            elif data[i, 0] < data[i, 1]:
                new_data.append(data[i, :])
            else:  # Zero duration (equality)
                print('Zero duration mark found and removed')
        new_data = np.stack(new_data, axis=0)

        raw_marks = data[:, [0, 1]]
        valid = data[:, 4]

        print('Loading %s' % path_eeg_file)
        with pyedflib.EdfReader(path_eeg_file) as file:
            signal = file.readSignal(0)
            signal_len = signal.shape[0]

        print('Starting correction... ', end='', flush=True)
        # Separate according to valid value. Valid = 0 is ignored.
        raw_marks_1 = raw_marks[valid == 1]
        raw_marks_2 = raw_marks[valid == 2]

        # Turn into binary sequence
        raw_marks_1 = data_ops.inter2seq(raw_marks_1, 0, signal_len - 1,
                                         allow_early_end=True)
        raw_marks_2 = data_ops.inter2seq(raw_marks_2, 0, signal_len - 1,
                                         allow_early_end=True)
        # Go back to intervals
        raw_marks_1 = data_ops.seq2inter(raw_marks_1)
        raw_marks_2 = data_ops.seq2inter(raw_marks_2)
        # In this way, overlapping intervals are now together

        # Correction rule:
        # Keep valid=2 always
        # Keep valid=1 only if there is no intersection with valid=2
        final_marks = [raw_marks_2]
        final_valid = [2 * np.ones(raw_marks_2.shape[0])]
        for i in range(raw_marks_1.shape[0]):
            # Check if there is any intersection
            add_condition = True
            for j in range(raw_marks_2.shape[0]):
                start_intersection = max(raw_marks_1[i, 0], raw_marks_2[j, 0])
                end_intersection = min(raw_marks_1[i, 1], raw_marks_2[j, 1])
                intersection = end_intersection - start_intersection
                if intersection >= 0:
                    add_condition = False
                    break
            if add_condition:
                final_marks.append(raw_marks_1[[i], :])
                final_valid.append([1])

        # Now concatenate everything
        final_marks = np.concatenate(final_marks, axis=0)
        final_valid = np.concatenate(final_valid, axis=0)

        # Now create array in right format
        # [start end -50 -50 valid channel]
        channel_for_txt = channel + 1
        number_for_txt = -50
        n_marks = final_marks.shape[0]
        channel_column = channel_for_txt * np.ones(n_marks).reshape([n_marks, 1])
        number_column = number_for_txt * np.ones(n_marks).reshape([n_marks, 1])
        valid_column = final_valid.reshape([n_marks, 1])
        table = np.concatenate(
            [final_marks,
             number_column, number_column,
             valid_column, channel_column],
            axis=1
        )
        table = table.astype(np.int32)

        # Now sort according to start time
        table = table[table[:, 0].argsort()]
        print('Done')

        # Now save into a file
        path_new_marks_file = os.path.abspath(os.path.join(
            inta_folder, PATH_MARKS, filename_format % name))
        np.savetxt(path_new_marks_file, table, fmt='%d', delimiter=' ')
        print('Fixed marks saved at %s\n' % path_new_marks_file)
