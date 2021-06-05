from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys
import pickle

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


def get_fit_id(config):
    fit_id = "%s_e%d_%s" % (config["dataset_name"], config["which_expert"], config["strategy"])
    return fit_id


def get_fitted_settings(source_config):
    fitting_save_dir = os.path.join(BASELINES_SAVE_PATH, BASELINE_FOLDER)
    fit_id = get_fit_id(source_config)
    fitted_setting_path = os.path.join(fitting_save_dir, 'fitted_%s.json' % fit_id)
    with open(fitted_setting_path, 'r') as infile:
        fitted_setting_dict = json.load(infile)
    print("Retrieving settings from %s" % fitted_setting_path, flush=True)
    return fitted_setting_dict


def predict(target_config, fitted_setting_dict):
    fs = 200
    dataset_short_name = target_config["dataset_name"].split("_")[0]
    pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'output_thesis_%s' % dataset_short_name)
    dataset = reader.load_dataset(target_config["dataset_name"], verbose=False, params={pkeys.FS: fs})
    _, _, test_ids_list = butils.get_partitions(dataset, target_config["strategy"], target_config["n_seeds"])
    print('Retrieving predictions from %s' % pred_dir, flush=True)
    pred_dict = butils.get_prediction_dict_from_settings(
        test_ids_list, dataset, pred_dir, fitted_setting_dict, get_raw_marks)
    return pred_dict


def save_predictions(pred_dict, filename):
    pred_save_dir = os.path.join(BASELINES_SAVE_PATH, BASELINE_FOLDER)
    with open(os.path.join(pred_save_dir, '%s.pkl' % filename), 'wb') as handle:
        pickle.dump(pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    fixed_configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1,
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2,
            strategy='fixed', n_seeds=11),
    ]
    cv_configs = [
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
    # Fixed configs are not involved in cross-dataset prediction
    for config in fixed_configs:
        prediction_id = "%s_from_%s" % (get_fit_id(config), get_fit_id(config))
        print("Predicting %s" % prediction_id)
        fitted_params = get_fitted_settings(config)
        predictions_dict = predict(config, fitted_params)
        save_predictions(predictions_dict, prediction_id)
        print("")

    for target_config in cv_configs:
        for source_config in cv_configs:
            prediction_id = "%s_from_%s" % (get_fit_id(target_config), get_fit_id(source_config))
            print("Predicting %s" % prediction_id)
            fitted_params = get_fitted_settings(source_config)
            predictions_dict = predict(target_config, fitted_params)
            save_predictions(predictions_dict, prediction_id)
            print("")
