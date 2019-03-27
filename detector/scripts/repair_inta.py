"""
Repairs INTA expert marks according to valid index.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import pyedflib

detector_path = '..'
sys.path.append(detector_path)

from sleep.inta import NAMES
from sleep import data_ops


if __name__ == '__main__':

    channel = 0  # f4-c4 channel

    filename_format = 'NewFixedSS_%s.txt'

    for name in NAMES:

        print('Fixing %s' % name)

        # Read marks
        path_marks_file = os.path.join(
            '..', '..', 'data', 'ssdata_inta', 'label', 'marks',
            'SS_%s.txt' % name)
        path_eeg_file = os.path.join(
            '..', '..', 'data', 'ssdata_inta', 'register',
            '%s.rec' % name)

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
                print('Zero duration found')
        new_data = np.stack(new_data, axis=0)
        print('data',data.shape)
        print('new_data',new_data.shape)

        raw_marks = data[:, [0, 1]]
        valid = data[:, 4]

        # Separate according to valid value
        # Valid = 0 is ignored.
        raw_marks_1 = raw_marks[valid == 1]
        raw_marks_2 = raw_marks[valid == 2]

        print('ra1', raw_marks_1.shape)
        print('ra2', raw_marks_2.shape)

        with pyedflib.EdfReader(path_eeg_file) as file:
            signal = file.readSignal(0)
            signal_len = signal.shape[0]
            print('len', signal_len)

        # Turn into binary sequence

        print('1 min, max', np.min(raw_marks_1), np.max(raw_marks_1))
        print('2 min, max', np.min(raw_marks_2), np.max(raw_marks_2))

        raw_marks_1 = data_ops.inter2seq(raw_marks_1, 0, signal_len - 1,
                                         allow_early_end=True)
        raw_marks_2 = data_ops.inter2seq(raw_marks_2, 0, signal_len - 1,
                                         allow_early_end=True)
        # Go back to intervals
        raw_marks_1 = data_ops.seq2inter(raw_marks_1)
        raw_marks_2 = data_ops.seq2inter(raw_marks_2)
        # In this way, overlapping intervals are now together

        print('ra1', raw_marks_1.shape)
        print('ra2', raw_marks_2.shape)

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

        print('final_marks', final_marks.shape)
        print('final_valid', final_valid.shape)

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
        print('table', table.shape)
        print(table[:6, :])
        print('')

        # TODO: Now sort according to start time
        # TODO: Now save into a file
        # TODO: Check visually the result in jupyter notebook

        path_new_marks_file = os.path.join(
            '..', '..', 'data', 'ssdata_inta', 'label', 'marks',
            filename_format % name)
        print('Saving at %s' % path_new_marks_file)
