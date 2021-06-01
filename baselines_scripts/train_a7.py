from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys

BASELINE_FOLDER = '2019_lacourse_a7'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))
BASELINES_SAVE_PATH = os.path.abspath(os.path.join(project_root, 'resources', 'comparison_data', 'baselines_2021'))


def extract_setting(fname):
    setting = "_".join(fname.split("_")[2:])[:-4]
    return setting


def get_raw_marks(predictions_dir, subject_id, setting, dataset):
    filepath = os.path.join(predictions_dir, 's%s' % subject_id, 'EventDetection_s%s_%s.txt' % (subject_id, setting))
    pred_data = pd.read_csv(filepath, sep='\t')
    # We substract 1 to translate from matlab to numpy indexing system
    start_samples = pred_data.start_sample.values - 1
    end_samples = pred_data.end_sample.values - 1
    # Now transform fs, preserving duration
    durations = end_samples - start_samples + 1
    fs_predicted = 100
    fs_target = dataset.fs
    start_samples = start_samples * fs_target / fs_predicted
    durations = durations * fs_target / fs_predicted
    end_samples = start_samples + durations - 1
    # Stack
    pred_marks = np.stack([start_samples, end_samples], axis=1).astype(np.int32)
    return pred_marks


if __name__ == '__main__':

    save_fitting = True

    fs = 200
    iou_threshold_report = 0.2
    # Dataset training settings
    configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1,
            strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2,
            strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1,
            strategy='5cv', n_seeds=3, average_mode=constants.MACRO_AVERAGE),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2,
            strategy='5cv', n_seeds=3, average_mode=constants.MACRO_AVERAGE),
    ]
    # ------------------------------------------------------
    for config in configs:
        dataset_name = config['dataset_name']
        which_expert = config['which_expert']
        strategy = config['strategy']  # {'fixed' or '5cv'}
        n_seeds = config['n_seeds']
        average_mode = config['average_mode']
        # Fit ID
        fit_id = "%s_e%d_%s" % (dataset_name, which_expert, strategy)
        print("\nEvaluation of %s (%s)" % (fit_id, average_mode))
        # Load dataset
        dataset = reader.load_dataset(dataset_name, verbose=True, params={pkeys.FS: fs})
        # Predictions
        pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'output_thesis_%s' % dataset_name.split("_")[0])
        print('Loading predictions from %s' % pred_dir, flush=True)
        settings = butils.get_settings(pred_dir, extract_setting)
        pred_dict = butils.get_prediction_dict(dataset, pred_dir, settings, get_raw_marks)
        train_ids_list, _, test_ids_list = butils.get_partitions(dataset, strategy, n_seeds)
        fitted_setting_dict = butils.train_grid(dataset, which_expert, train_ids_list, pred_dict, average_mode)

        if save_fitting:
            fitting_save_dir = os.path.join(BASELINES_SAVE_PATH, BASELINE_FOLDER)
            os.makedirs(fitting_save_dir, exist_ok=True)
            fname = os.path.join(fitting_save_dir, 'fitted_%s.json' % fit_id)
            with open(fname, 'w') as outfile:
                json.dump(fitted_setting_dict, outfile)

        results = butils.evaluate_by_fold(
            dataset, which_expert, test_ids_list, fitted_setting_dict, pred_dict, average_mode, iou_threshold_report)
        print("\nTest performance (%s, iou >= %1.2f, from train to test) for %s" % (
            average_mode, iou_threshold_report, fit_id))
        butils.print_performance(results)
        del dataset
