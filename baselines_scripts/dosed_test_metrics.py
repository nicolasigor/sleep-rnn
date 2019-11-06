from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint
import pickle

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.data import utils, stamp_correction
from sleeprnn.detection import metrics, postprocessing
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(
    project_root, '../sleep-baselines/2019_chambon_dosed'))

if __name__ == '__main__':
    algorithm_name = '2019_chambon'

    dataset_name = constants.MASS_KC_NAME
    which_expert = 1
    fs = 128
    dataset_params = {pkeys.FS: fs}

    task_mode = constants.N2_RECORD
    id_try_list = np.arange(10)

    # Map from dataset to folder
    dataset_to_folder = {
        (constants.MASS_SS_NAME, 1): '20191101_10fold_n2D_spindle1_stdNorm',
        (constants.MASS_SS_NAME, 2): '20191101_10fold_n2D_spindle2_stdNorm',
        (constants.MASS_KC_NAME, 1): '20191101_10fold_n2D_kcomplex_stdFix',
    }

    # Load expert annotations
    dataset = load_dataset(
        dataset_name,
        params=dataset_params)
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

    # for kc
    test_signals_list = dataset.get_subset_signals(test_ids)

    # Load predictions
    pred_fold_dict = {}
    setting_dict = {}
    for k in id_try_list:
        print('Processing fold %d' % k, flush=True)
        pred_folder = os.path.join(
            BASELINE_PATH, 'results', dataset_to_folder[(dataset_name, which_expert)])
        print('Loading predictions from %s' % pred_folder, flush=True)
        pred_file = os.path.join(pred_folder, 'seed%d_predictions_ckpt.pkl' % k)
        thr_file = os.path.join(pred_folder, 'seed%d_dosed_threshold_ckpt.pkl' % k)
        with open(pred_file, 'rb') as handle:
            this_predictions = pickle.load(handle)
        with open(thr_file, 'rb') as handle:
            this_thr = pickle.load(handle)
        setting = 'thr(%1.2f)' % this_thr
        setting_dict[k] = setting
        pred_fold_dict[k] = {}
        for record_name in this_predictions.keys():
            subject_id = int(record_name.split(" ")[0].split("-")[-1])
            print('Subject %d, setting %s' % (subject_id, setting), flush=True)
            pred_marks = np.array(this_predictions[record_name][0])
            pred_marks = (pred_marks * fs / 256).astype(np.int32)

            # Valid subset of marks
            pred_marks_n2 = utils.extract_pages_for_stamps(
                pred_marks, n2_dict[subject_id], page_size)

            # Postprocessing
            if dataset_name == constants.MASS_SS_NAME:
                pred_marks_n2 = stamp_correction.combine_close_stamps(
                    pred_marks_n2, fs, min_separation=0.3)
                pred_marks_n2 = stamp_correction.filter_duration_stamps(
                    pred_marks_n2, fs, min_duration=0.25, max_duration=3.0)
            elif dataset_name == constants.MASS_KC_NAME:
                pred_marks_n2 = stamp_correction.filter_duration_stamps(
                    pred_marks_n2, fs, min_duration=0.3, max_duration=None)
                idx_subject = test_ids.index(subject_id)
                this_signal = test_signals_list[idx_subject]
                pred_marks_n2 = postprocessing.kcomplex_stamp_split(
                    this_signal, pred_marks_n2, fs)
            else:
                raise ValueError('Dataset name invalid')

            # Save marks for evaluation
            pred_fold_dict[k][subject_id] = pred_marks_n2

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
