from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint
import json

import numpy as np
from scipy.io import loadmat

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.data import utils, stamp_correction
from sleeprnn.detection import metrics
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(
    project_root, '../sleep-baselines/2017_lajnef_spinky'))

if __name__ == '__main__':

    dataset_name = constants.MASS_KC_NAME
    which_expert = 1
    fs = 128

    task_mode = constants.N2_RECORD
    id_try_list = np.arange(10)

    # Load expert annotations
    dataset = load_dataset(
        dataset_name, load_checkpoint=True, params={pkeys.FS: fs})

    # ----------------------------------------------------------------------
    # CROSS - VALIDATION
    # ----------------------------------------------------------------------

    all_train_ids = dataset.train_ids
    page_size = dataset.page_size
    print('Page size:', page_size)
    print('All train ids', all_train_ids)
    gs_dict = {}
    n2_dict = {}
    for subject_id in all_train_ids:
        n2_dict[subject_id] = dataset.get_subject_pages(
            subject_id,
            pages_subset=task_mode)

    # Load predictions
    context_size = int(5 * fs)
    pred_folder = os.path.join(BASELINE_PATH, 'output_%s' % dataset_name)
    print('Loading predictions from %s' % pred_folder, flush=True)
    pred_files = os.listdir(pred_folder)
    pred_dict = {}
    visited_settings = []
    for file in pred_files:
        subject_id = int(file.split('_')[1][1:])
        # only subjects for cross-validation
        if subject_id not in all_train_ids:
            continue

        setting = file.split('_')[2][:-4]
        if setting not in visited_settings:
            pred_dict[setting] = {}
            visited_settings.append(setting)

        # Binary sequences
        filepath = os.path.join(pred_folder, file)
        pred_data = loadmat(filepath)
        pred_data = pred_data['detection_matrix']

        # Now we need to do different things if SS or KC
        if dataset_name == constants.MASS_SS_NAME:

            # Spindles
            start_sample = context_size
            end_sample = pred_data.shape[1] - context_size - 1
            pred_marks_list = []
            for page_idx in range(pred_data.shape[0]):
                page_sequence = pred_data[page_idx, :]
                page_marks = utils.seq2stamp(page_sequence)
                # Now we keep marks that at least partially contained in page
                if page_marks.size > 0:
                    page_marks = utils.filter_stamps(page_marks, start_sample, end_sample)
                    if page_marks.size > 0:
                        # Before we append them, we need to provide the right
                        # sample
                        real_page = n2_dict[subject_id][page_idx]
                        offset_sample = int(real_page * page_size - context_size)
                        page_marks = page_marks + offset_sample
                        pred_marks_list.append(page_marks)
            if pred_marks_list:
                pred_marks = np.concatenate(pred_marks_list, axis=0)
                # Now we need to post-process
                pred_marks = stamp_correction.combine_close_stamps(
                    pred_marks, fs, min_separation=0.3)
                pred_marks = stamp_correction.filter_duration_stamps(
                    pred_marks, fs, min_duration=0.3, max_duration=3.0)
            else:
                pred_marks = np.array([]).reshape((0, 2))
            pred_dict[setting][subject_id] = pred_marks

        elif dataset_name == constants.MASS_KC_NAME:
            # KC
            # We only keep what's inside the page
            pred_data = pred_data[:, context_size:-context_size]
            # Now we concatenate and then the extract stamps
            pred_marks = utils.seq2stamp_with_pages(
                pred_data, n2_dict[subject_id])
            #  Now we manually add 0.1 s before and 1.3 s after (paper)
            add_before = int(np.round(0.1 * fs))
            add_after = int(np.round(1.3 * fs))
            pred_marks[:, 0] = pred_marks[:, 0] - add_before
            pred_marks[:, 1] = pred_marks[:, 1] + add_after
            # By construction all marks are valid (inside N2 pages)
            # Save marks for evaluation
            pred_dict[setting][subject_id] = pred_marks
        else:
            raise ValueError('Invalid dataset_name')

    # Start evaluation
    print('Starting evaluation of %d settings' % len(visited_settings), flush=True)
    cross_val_results = {}
    for k in id_try_list:
        print('Using fold %d. ' % k, flush=True, end='')
        train_ids, _ = utils.split_ids_list_v2(all_train_ids, split_id=k)
        train_ids.sort()
        train_marks_list = dataset.get_subset_stamps(
            train_ids,
            which_expert=which_expert,
            pages_subset=task_mode)
        setting_performance = []
        for setting in visited_settings:
            # Create subset of marks
            pred_marks_list = []
            for subject_id in train_ids:
                pred_marks_list.append(pred_dict[setting][subject_id])
            af1_of_setting = metrics.average_metric_with_list(
                train_marks_list, pred_marks_list, verbose=False)
            setting_performance.append(af1_of_setting)

        # Look for best performance
        max_idx = np.argmax(setting_performance).item()
        max_af1 = setting_performance[max_idx]
        max_setting = visited_settings[max_idx]
        print('Best AF1 %1.4f with setting %s' % (max_af1, max_setting), flush=True)
        cross_val_results['fold%d' % k] = max_setting

    # Save crossval results
    fname = '2017_lajnef_xval_%s_e%d.json' % (dataset_name, which_expert)
    with open(fname, 'w') as handle:
        json.dump(cross_val_results, handle)
