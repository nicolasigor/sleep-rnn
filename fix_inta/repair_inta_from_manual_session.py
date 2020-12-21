from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.inta_ss import NAMES
from sleeprnn.data import utils


def get_mark_from_idx(idx_mark, raw_1, raw_2):
    if idx_mark < 2e4:
        # It is 10000 series
        idx_mark = int(idx_mark - 1e4)
        single_mark = raw_1[idx_mark]
    else:
        # It is 20000 series
        idx_mark = int(idx_mark - 2e4)
        single_mark = raw_2[idx_mark]
    return single_mark


if __name__ == '__main__':

    # IMPORTANT:
    # This correction strategy assumes that all conflicting overlaps
    # (i.e. group of marks not represented by automatically accepted ones)
    # were solved during the manual session.
    # Therefore, the correction is as follows:
    # 1. Add all manually accepted marks.
    # 2. Add green marks (previously accepted marks) only if
    # they do not intersect with manually rejected or accepted ones.
    #
    # In consequence, all remaining conflicts are implicitly rejected.
    # This behavior produces unwanted results if the manual session is
    # incomplete. Please do not use the corrected marks generated for an
    # incomplete manual session as the new labels.

    subject_id = 2

    subject_name = NAMES[subject_id - 1]
    print('Processing %s' % subject_name)
    fs = 200
    raw_stamps_1 = np.loadtxt('mark_files/%s_raw_1.txt' % subject_name).astype(np.int32)
    raw_stamps_2 = np.loadtxt('mark_files/%s_raw_2.txt' % subject_name).astype(np.int32)
    marks_without_doubt = np.loadtxt('mark_files/%s_without_doubt.txt' % subject_name).astype(np.int32)

    # print(raw_stamps_1)
    # print(raw_stamps_2)
    # print(marks_without_doubt)

    accepted_df = pd.read_csv('mark_files/%s_garrido_accepted_20190805.csv' % subject_name)
    print(accepted_df)

    rejected_df = pd.read_csv(
        'mark_files/%s_garrido_rejected_20190805.csv' % subject_name)
    print(rejected_df)

    # Transform marks from df to the standard [start end] format

    def my_func(single_row):

        idx_start = single_row['idx_start']
        idx_end = single_row['idx_end']
        dt_start = single_row['dt_start']
        dt_end = single_row['dt_end']

        t_start = single_row['t_start']
        t_end = single_row['t_end']

        if np.isnan(t_start):
            # We need the start time
            this_mark = get_mark_from_idx(idx_start, raw_stamps_1, raw_stamps_2)
            sample_start = this_mark[0]
            if not np.isnan(dt_start):
                sample_start = int(sample_start - dt_start * fs)
        else:
            sample_start = int(t_start * fs)

        if np.isnan(t_end):
            # We need the end time
            if np.isnan(idx_end):
                idx_end = idx_start
            this_mark = get_mark_from_idx(idx_end, raw_stamps_1, raw_stamps_2)
            sample_end = this_mark[1]
            if not np.isnan(dt_end):
                sample_end = int(sample_end + dt_end * fs)
        else:
            sample_end = int(t_end * fs)

        new_row = pd.Series([sample_start, sample_end], index=['start', 'end'])
        return new_row


    accepted_df = accepted_df.apply(my_func, axis=1)

    def my_func_rejected(single_row):
        idx_mark = single_row['idx']
        this_mark = get_mark_from_idx(idx_mark, raw_stamps_1, raw_stamps_2)
        new_row = pd.Series([this_mark[0], this_mark[1]], index=['start', 'end'])
        return new_row


    rejected_df = rejected_df.apply(my_func_rejected, axis=1)

    # Recover only the matrix:
    accepted_marks = accepted_df.to_numpy()
    rejected_marks = rejected_df.to_numpy()

    print('Accepted marks:', accepted_marks.shape)
    print('Rejected marks:', rejected_marks.shape)

    # Check overlap of accepted marks
    overlap_check = utils.get_overlap_matrix(
        accepted_marks, accepted_marks)
    overlap_check = overlap_check.sum(axis=1) - 1
    if np.any(overlap_check):
        print(np.where(overlap_check != 0)[0])
        print(accepted_marks[np.where(overlap_check != 0)[0], :])
        print(accepted_marks[np.where(overlap_check != 0)[0], :] / fs)
        raise ValueError('Manually accepted marks are overlapped.')

    # Now everything is in the right format
    # Start process of building the new mark file

    # First we accept all accepted marks immediately, with a valid=3

    valid_for_new = 3 * np.ones(shape=(accepted_marks.shape[0], 1))
    accepted_marks_with_valid = np.concatenate(
        [accepted_marks, valid_for_new], axis=1)

    # Now, we preprocess the previously added marks removing the rejected ones
    overlap_between_green_and_rejected = utils.get_overlap_matrix(
        marks_without_doubt[:, [0, 1]], rejected_marks)

    exists_overlap = overlap_between_green_and_rejected.sum(axis=1)
    valid_marks = np.where(exists_overlap == 0)[0]

    marks_without_doubt_corrected = marks_without_doubt[valid_marks, :]

    # Now, add the green marks that do not intersect with the accepted marks
    overlap_between_green_and_accepted = utils.get_overlap_matrix(
        marks_without_doubt_corrected[:, [0, 1]], accepted_marks)

    exists_overlap = overlap_between_green_and_accepted.sum(axis=1)
    valid_marks = np.where(exists_overlap == 0)[0]

    marks_without_doubt_corrected = marks_without_doubt_corrected[valid_marks, :]

    corrected_marks = np.concatenate(
        [accepted_marks_with_valid, marks_without_doubt_corrected], axis=0)

    corrected_marks = corrected_marks[np.argsort(corrected_marks[:, 0]), :]

    # Add stuff to maintain compatibility
    channel_for_txt = 1
    number_for_txt = -50
    final_marks = corrected_marks[:, [0, 1]]
    n_marks = final_marks.shape[0]
    valid_column = corrected_marks[:, 2].reshape([n_marks, 1])

    channel_column = channel_for_txt * np.ones(n_marks).reshape(
        [n_marks, 1])
    number_column = number_for_txt * np.ones(n_marks).reshape(
        [n_marks, 1])

    table = np.concatenate(
        [final_marks,
         number_column, number_column,
         valid_column, channel_column],
        axis=1
    )
    table = table.astype(np.int32)

    # Check non-overlapping marks
    overlap_check = utils.get_overlap_matrix(
        final_marks, final_marks)
    overlap_check = overlap_check.sum(axis=1) - 1
    if np.any(overlap_check):
        raise ValueError('Final marks are overlapped.')
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    fname = 'mark_files/%s_Revision_SS_%s.txt' % (this_date, subject_name)
    np.savetxt(fname, table, fmt='%i')
    print("Revision saved at %s" % fname)
