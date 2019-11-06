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

from sleeprnn.data.loader import load_dataset
from sleeprnn.data import utils, stamp_correction
from sleeprnn.detection import metrics
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(
    project_root, '../sleep-baselines/2017_lajnef_spinky'))


if __name__ == '__main__':
    algorithm_name = '2017_lajnef'

    dataset_name = constants.MASS_SS_NAME
    which_expert = 2
    fs = 128
    dataset_params = {pkeys.FS: fs}

    task_mode = constants.N2_RECORD
    id_try_list = np.arange(10)

    # Load expert annotations
    dataset = load_dataset(
        dataset_name, load_checkpoint=True, params={pkeys.FS: fs})

    test_ids = dataset.test_ids
    page_size = dataset.page_size
    print('Page size:', page_size)
    print('Test ids', test_ids)
    gs_dict = {}
    n2_dict = {}
    for subject_id in test_ids:
        n2_dict[subject_id] = dataset.get_subject_pages(
            subject_id,
            pages_subset=task_mode)

    # Load xval results
    fname = '%s_xval_%s_e%d.json' % (algorithm_name, dataset_name, which_expert)
    with open(fname, 'r') as handle:
        xval_dict = json.load(handle)

    # Load predictions
    context_size = int(5 * fs)
    pred_folder = os.path.join(BASELINE_PATH, 'output_%s' % dataset_name)
    print('Loading predictions from %s' % pred_folder, flush=True)
    pred_files = os.listdir(pred_folder)

    pred_fold_dict = {}
    setting_dict = {}
    for k in id_try_list:
        print('Processing fold %d' % k, flush=True)
        target_setting = xval_dict['fold%d' % k]
        print('Target setting: %s' % target_setting)
        setting_dict[k] = target_setting
        pred_fold_dict[k] = {}
        for file in pred_files:
            subject_id = int(file.split('_')[1][1:])
            setting = file.split('_')[2][:-4]
            if subject_id not in test_ids:
                continue
            if setting != target_setting:
                continue
            # Now we have the right subject and the right setting
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
                        page_marks = utils.filter_stamps(page_marks,
                                                         start_sample,
                                                         end_sample)
                        if page_marks.size > 0:
                            # Before we append them, we need to provide the right
                            # sample
                            real_page = n2_dict[subject_id][page_idx]
                            offset_sample = int(
                                real_page * page_size - context_size)
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
                pred_fold_dict[k][subject_id] = pred_marks

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
                pred_fold_dict[k][subject_id] = pred_marks
            else:
                raise ValueError('Invalid dataset_name')

    # Start evaluation
    print('\nStarting evaluation\n', flush=True)
    iou_axis = np.linspace(0.05, 0.95, 19)
    test_marks_list = dataset.get_subset_stamps(
        test_ids,
        which_expert=which_expert,
        pages_subset=task_mode)
    mean_af1_list = []
    mean_iou_list = []
    for k in id_try_list:
        # print('Using fold %d' % k, flush=True)
        # Evaluate each subject separately
        af1_list = []
        iou_list = []
        for i, subject_id in enumerate(test_ids):
            this_gs = test_marks_list[i]
            this_det = pred_fold_dict[k][subject_id]

            # matching
            iou_matching, idx_array = metrics.matching(
                this_gs, this_det)

            # curves
            f1_iou = metrics.metric_vs_iou(
                this_gs, this_det, iou_axis, metric_name=constants.F1_SCORE,
                iou_matching=iou_matching)
            precision_iou = metrics.metric_vs_iou(
                this_gs, this_det, iou_axis, metric_name=constants.PRECISION,
                iou_matching=iou_matching)
            recall_iou = metrics.metric_vs_iou(
                this_gs, this_det, iou_axis, metric_name=constants.RECALL,
                iou_matching=iou_matching)

            # scalars
            subject_af1 = metrics.average_metric(
                this_gs, this_det, iou_matching=iou_matching)
            subject_iou = np.mean(iou_matching[idx_array > -1])

            print('Fold %d, S%02d: Test AF1 %1.2f -- Test IoU %1.2f'
                  % (k, subject_id, 100 * subject_af1, 100 * subject_iou), flush=True)
            af1_list.append(subject_af1)
            iou_list.append(subject_iou)

            # Save results
            predictions_seconds = this_det / fs
            folder_to_save = '%s/%s/e%d/fold%d' % (algorithm_name, dataset_name, which_expert, k)
            os.makedirs(folder_to_save, exist_ok=True)
            filename = '%s_%s_e%d_fold%d_s%02d.npz' % (
                algorithm_name, dataset_name, which_expert, k, subject_id)
            np.savez(
                os.path.join(folder_to_save, filename),
                algorithm_id=algorithm_name,
                dataset_id=dataset_name,
                expert_id=which_expert,
                fold_id=k,
                subject_id=subject_id,
                parameters=setting_dict[k],
                subject_predicted_marks=predictions_seconds.astype(np.float32),
                subject_af1=subject_af1.astype(np.float32),
                subject_iou=subject_iou.astype(np.float32),
                iou_axis=iou_axis.astype(np.float32),
                f1_vs_iou=f1_iou.astype(np.float32),
                precision_vs_iou=precision_iou.astype(np.float32),
                recall_vs_iou=recall_iou.astype(np.float32)
            )

        mean_af1 = np.mean(af1_list)
        mean_iou = np.mean(iou_list)
        print('Fold final stats: AF1 %1.2f -- IoU %1.2f \n' % (
            100 * mean_af1, 100 * mean_iou
        ))
        mean_af1_list.append(mean_af1)
        mean_iou_list.append(mean_iou)

    print('\nTest set final stats: AF1 %1.2f/%1.2f -- IoU %1.2f/%1.2f' % (
        100 * np.mean(mean_af1_list), 100 * np.std(mean_af1_list),
        100 * np.mean(mean_iou_list), 100 * np.std(mean_iou_list)
    ))
