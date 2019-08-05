from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.utils import PATH_DATA
from sleeprnn.data.inta_ss import PATH_INTA_RELATIVE, PATH_REC, PATH_MARKS, NAMES
from sleeprnn.data import utils
from sleeprnn.detection import metrics
from sleeprnn.helpers import reader


if __name__ == '__main__':
    # Load stamps of subject
    for subject_id in range(1, 12):
        print('Loading S%02d' % subject_id)
        fs = 200
        dataset_dir = os.path.join(PATH_DATA, PATH_INTA_RELATIVE)
        path_stamps = os.path.join(
            dataset_dir, PATH_MARKS, 'SS_%s.txt' % NAMES[subject_id - 1])
        path_signals = os.path.join(
            dataset_dir, PATH_REC, '%s.rec' % NAMES[subject_id - 1])

        raw_stamps_1, raw_stamps_2 = reader.load_raw_inta_stamps(
            path_stamps, path_signals, min_samples=20, chn_idx=0)

        durations_1 = (raw_stamps_1[:, 1] - raw_stamps_1[:, 0]) / fs
        durations_2 = (raw_stamps_2[:, 1] - raw_stamps_2[:, 0]) / fs
        print('V1', raw_stamps_1.shape, 'Min dur [s]', durations_1.min(),
              'Max dur [s]', durations_1.max())
        print('V2', raw_stamps_2.shape, 'Min dur [s]', durations_2.min(),
              'Max dur [s]', durations_2.max())

        overlap_m = utils.get_overlap_matrix(raw_stamps_1, raw_stamps_1)
        groups_overlap_1 = utils.overlapping_groups(overlap_m)
        overlap_m = utils.get_overlap_matrix(raw_stamps_2, raw_stamps_2)
        groups_overlap_2 = utils.overlapping_groups(overlap_m)

        n_overlaps_1 = [len(single_group) for single_group in groups_overlap_1]
        values_1, counts_1 = np.unique(n_overlaps_1, return_counts=True)
        print('\nSize of overlapping groups for Valid 1')
        for value, count in zip(values_1, counts_1):
            print('%d marks: %d times' % (value, count))

        n_overlaps_2 = [len(single_group) for single_group in groups_overlap_2]
        values_2, counts_2 = np.unique(n_overlaps_2, return_counts=True)
        print('\nSize of overlapping groups for Valid 2')
        for value, count in zip(values_2, counts_2):
            print('%d marks: %d times' % (value, count))

        max_overlaps = np.max([values_1.max(), values_2.max()]) - 1

        # #### Conflicts

        # Select marks without doubt
        groups_in_doubt_v1_list = []
        groups_in_doubt_v2_list = []

        iou_to_accept = 0.8
        marks_without_doubt = []
        valid_without_doubt = []
        overlap_between_1_and_2 = utils.get_overlap_matrix(
            raw_stamps_1, raw_stamps_2)

        for single_group in groups_overlap_2:
            if len(single_group) == 1:
                marks_without_doubt.append(raw_stamps_2[single_group[0], :])
                valid_without_doubt.append(2)
            elif len(single_group) == 2:
                # check if IOU between marks is close 1, if close, then just choose newer (second one)
                option1_mark = raw_stamps_2[single_group[0], :]
                option2_mark = raw_stamps_2[single_group[1], :]
                iou_between_marks = metrics.get_iou(option1_mark, option2_mark)
                if iou_between_marks >= iou_to_accept:
                    marks_without_doubt.append(option2_mark)
                    valid_without_doubt.append(2)
                else:
                    groups_in_doubt_v2_list.append(single_group)
            else:
                groups_in_doubt_v2_list.append(single_group)

        for single_group in groups_overlap_1:
            is_in_doubt = False
            # Check if entire group is overlapping
            all_are_overlapping_2 = np.all(
                overlap_between_1_and_2[single_group, :].sum(axis=1))
            if not all_are_overlapping_2:
                # Consider the mark
                if len(single_group) == 1:
                    # Since has size 1 and is no overlapping 2, accept it
                    marks_without_doubt.append(raw_stamps_1[single_group[0], :])
                    valid_without_doubt.append(1)
                elif len(single_group) == 2:
                    # check if IOU between marks is close 1, if close, then just choose newer (second one) since there is no intersection
                    option1_mark = raw_stamps_1[single_group[0], :]
                    option2_mark = raw_stamps_1[single_group[1], :]
                    iou_between_marks = metrics.get_iou(option1_mark, option2_mark)
                    if iou_between_marks >= iou_to_accept:
                        marks_without_doubt.append(raw_stamps_1[single_group[1], :])
                        valid_without_doubt.append(1)
                    else:
                        is_in_doubt = True
                else:
                    is_in_doubt = True
            if is_in_doubt:
                groups_in_doubt_v1_list.append(single_group)

        marks_without_doubt = np.stack(marks_without_doubt, axis=0).reshape((-1, 2))
        valid_without_doubt = np.stack(valid_without_doubt, axis=0).reshape((-1, 1))
        marks_without_doubt = np.concatenate([marks_without_doubt, valid_without_doubt], axis=1)
        marks_without_doubt = marks_without_doubt[marks_without_doubt[:, 0].argsort()]

        print('Marks automatically added:', marks_without_doubt.shape)
        print('Remaining conflicts:')
        print('    V1: %d' % len(groups_in_doubt_v1_list))
        print('    V2: %d' % len(groups_in_doubt_v2_list))

        # Now we have three marks subsets from the inta revision
        # marks_without_doubt: The green ones
        # raw_stamps_1: The 10000 series
        # raw_stamps_2: The 20000 series

        # ### Save

        np.savetxt('mark_files/%s_raw_1.txt' % NAMES[subject_id - 1], raw_stamps_1)
        np.savetxt('mark_files/%s_raw_2.txt' % NAMES[subject_id - 1], raw_stamps_2)
        np.savetxt('mark_files/%s_without_doubt.txt' % NAMES[subject_id - 1], marks_without_doubt)
