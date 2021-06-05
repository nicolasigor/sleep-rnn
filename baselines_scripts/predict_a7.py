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
    fs = 200
    fitting_save_dir = os.path.join(BASELINES_SAVE_PATH, BASELINE_FOLDER)
    configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1,
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2,
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1,
            strategy='5cv', n_seeds=3),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2,
            strategy='5cv', n_seeds=3),
        dict(
            dataset_name=constants.MODA_SS_NAME, which_expert=1,
            strategy='5cv', n_seeds=3),
        dict(
            dataset_name=constants.INTA_SS_NAME, which_expert=1,
            strategy='5cv', n_seeds=3),
    ]
    for target_config in configs:
        for source_config in configs:
            target_strategy = target_config['strategy']
            source_strategy = source_config['strategy']
            if target_strategy == 'fixed':
                print("PENDIENTE")

    #         dataset_name = config['dataset_name']
    #         which_expert = config['which_expert']
    #         strategy = config['strategy']  # {'fixed' or '5cv'}
    #         n_seeds = config['n_seeds']
    #         fit_id = "%s_e%d_%s" % (dataset_name, which_expert, strategy)
    #         print("\nPredictions of %s" % fit_id)
    #         pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'output_thesis_%s' % dataset_name.split("_")[0])
    #         fitted_setting_path = os.path.join(fitting_save_dir, 'fitted_%s.json' % fit_id)
    #         dataset = reader.load_dataset(dataset_name, verbose=True, params={pkeys.FS: fs})
    #         _, _, test_ids_list = butils.get_partitions(dataset, strategy, n_seeds)
    #         with open(fitted_setting_path, 'r') as infile:
    #             fitted_setting_dict = json.load(infile)
    #         print('Loading predictions from %s\nusing settings from %s' % (pred_dir, fitted_setting_path), flush=True)
    #         pred_dict = butils.get_prediction_dict_from_settings(
    #             test_ids_list, dataset, pred_dir, fitted_setting_dict, get_raw_marks)
    #         del dataset
    #
    #
    # # Cross dataset only for 5cv