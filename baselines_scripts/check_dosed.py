from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
from pprint import pprint
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys

BASELINE_FOLDER = '2019_chambon_dosed'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))


def get_raw_dosed_marks_in_fold(predictions_dir, fold_id, dataset):
    pred_filepath = os.path.join(predictions_dir, 'fold%02d_dosed_predictions_ckpt.pkl' % fold_id)
    thr_filepath = os.path.join(predictions_dir, 'fold%02d_dosed_threshold_ckpt.pkl' % fold_id)
    with open(pred_filepath, 'rb') as handle:
        this_predictions = pickle.load(handle)
    with open(thr_filepath, 'rb') as handle:
        this_thr = pickle.load(handle)
    print("Fold %02d with threshold %1.2f" % (fold_id, this_thr))
    records = list(this_predictions.keys())
    fs_predicted = int(records[0].split(".")[0].split("_")[3])
    fs_target = dataset.fs
    pred_dict = {}
    for record_name in records:
        subject_id = record_name.split(".")[0].split("_")[1][1:]
        dataset_name_short = record_name.split(".")[0].split("_")[0]
        if dataset_name_short in ['mass', 'inta']:
            subject_id = int(subject_id)
        pred_marks = np.array(this_predictions[record_name][0])
        pred_marks = (pred_marks * fs_target / fs_predicted).astype(np.int32)
        pred_dict[subject_id] = pred_marks
    return pred_dict


def get_raw_dosed_marks(predictions_dir, n_folds, dataset):
    pred_dict = {}
    for fold_id in range(n_folds):
        pred_dict[fold_id] = get_raw_dosed_marks(predictions_dir, fold_id, dataset)
    return pred_dict


def postprocess_marks_dosed(dataset, prediction_dict):
    kc_split = (dataset.event_name == constants.KCOMPLEX)
    new_prediction_dict = {}
    for fold_id in prediction_dict.keys():
        new_prediction_dict[fold_id] = {}
        for subject_id in prediction_dict[fold_id].keys():
            new_prediction_dict[fold_id][subject_id] = butils.postprocess_marks(
                dataset, prediction_dict[fold_id][subject_id], subject_id, kc_split=kc_split)
    return new_prediction_dict


if __name__ == '__main__':
    # TODO: postprocesar kcomplex split

    fs = 200
    iou_threshold_report = 0.2
    # Dataset training settings
    configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1,
            result_folder="20210603_thesis_mass_fixed_spindle1",
            strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1,
            result_folder="20210603_thesis_mass_fixed_kcomplex",
            strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
    ]
    # ------------------------------------------------------
    for config in configs:
        dataset_name = config['dataset_name']
        which_expert = config['which_expert']
        strategy = config['strategy']  # {'fixed' or '5cv'}
        n_seeds = config['n_seeds']
        average_mode = config['average_mode']
        result_folder = config['result_folder']
        # Fit ID
        fit_id = "%s_e%d_%s" % (dataset_name, which_expert, strategy)
        print("\nEvaluation of %s (%s)" % (fit_id, average_mode))
        # Load dataset
        dataset = reader.load_dataset(dataset_name, verbose=True, params={pkeys.FS: fs})
        # Predictions
        pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'results', result_folder)
        print('Loading predictions from %s' % pred_dir, flush=True)
        _, _, test_ids_list = butils.get_partitions(dataset, strategy, n_seeds)
        n_folds = len(test_ids_list)
        pred_dict = get_raw_dosed_marks(pred_dir, n_folds, dataset)
        pred_dict = postprocess_marks_dosed(dataset, pred_dict)


        # results = butils.evaluate_by_fold(
        #     dataset, which_expert, test_ids_list, fitted_setting_dict, pred_dict, average_mode, iou_threshold_report)
        # print("\nTest performance (%s, iou >= %1.2f, from train to test) for %s" % (
        #     average_mode, iou_threshold_report, fit_id))
        # butils.print_performance(results)
        # del dataset