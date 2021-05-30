from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants

BASELINE_FOLDER = '2019_lacourse_a7'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))
BASELINES_SAVE_PATH = os.path.abspath(os.path.join(project_root, 'resources', 'comparison_data', 'baselines_2021'))


def extract_setting(fname):
    setting = "_".join(fname.split("_")[2:])[:-4]
    return setting


def get_raw_marks(predictions_dir, subject_id, setting):
    filepath = os.path.join(predictions_dir, 's%s' % subject_id, 'EventDetection_s%s_%s.txt' % (subject_id, setting))
    pred_data = pd.read_csv(filepath, sep='\t')
    # We substract 1 to translate from matlab to numpy indexing system
    start_samples = pred_data.start_sample.values - 1
    end_samples = pred_data.end_sample.values - 1
    pred_marks = np.stack([start_samples, end_samples], axis=1)
    return pred_marks


if __name__ == '__main__':

    # TODO: generate script for final npz prediction files (indataset and crossdataset), to simplify results

    # Dataset training settings
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    strategy = 'fixed'  # {'fixed' or '5cv'}
    n_seeds = 11
    average_mode = constants.MACRO_AVERAGE
    iou_threshold_report = 0.2

    # ------------------------------------------------------

    # Load dataset
    dataset = reader.load_dataset(dataset_name, verbose=False)
    # Predictions
    pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'output_thesis_%s' % dataset_name.split("_")[0])
    print('Loading predictions from %s' % pred_dir, flush=True)
    settings = butils.get_settings(pred_dir, extract_setting)

    pred_dict = butils.get_prediction_dict(dataset, pred_dir, settings, get_raw_marks)
    train_ids_list, _, test_ids_list = butils.get_partitions(dataset, strategy, n_seeds)
    fitted_setting_dict = butils.train_grid(dataset, which_expert, train_ids_list, pred_dict, average_mode)
    fit_id = "%s_e%d" % (dataset_name, which_expert)

    # Save training
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
