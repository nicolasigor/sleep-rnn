from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys
import pickle

import numpy as np
from scipy.io import loadmat

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys
from sleeprnn.data import utils

BASELINE_FOLDER = '2017_lajnef_spinky'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))
BASELINES_SAVE_PATH = os.path.abspath(os.path.join(project_root, 'resources', 'comparison_data', 'baselines_2021'))


def extract_setting(fname):
    setting = fname.split("_")[2][:-4]
    return setting


def get_raw_marks(predictions_dir, subject_id, setting, dataset):
    filepath = os.path.join(predictions_dir, 's%s' % subject_id, 'kcomplex_s%s_%s.mat' % (subject_id, setting))
    pred_data = loadmat(filepath)
    pred_data = pred_data['detection_matrix']
    # KC
    # We only keep what's inside the page
    fs_predicted = 100
    context_size = int(5 * fs_predicted)
    pred_data = pred_data[:, context_size:-context_size]
    # Now we concatenate and then the extract stamps
    n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
    pred_marks = utils.seq2stamp_with_pages(pred_data, n2_pages)
    # Now transform to target fs
    fs_target = dataset.fs
    pred_marks = (pred_marks * fs_target / fs_predicted).astype(np.int32)
    #  Now we manually add 0.1 s before and 1.3 s after (paper)
    add_before = int(np.round(0.1 * fs_target))
    add_after = int(np.round(1.3 * fs_target))
    pred_marks[:, 0] = pred_marks[:, 0] - add_before
    pred_marks[:, 1] = pred_marks[:, 1] + add_after
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

    configs = [
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1,
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1,
            strategy='5cv', n_seeds=3),
    ]
    # Spinky is not involved in cross-dataset prediction
    for config in configs:
        prediction_id = "%s_from_%s" % (get_fit_id(config), get_fit_id(config))
        print("Predicting %s" % prediction_id)
        fitted_params = get_fitted_settings(config)
        predictions_dict = predict(config, fitted_params)
        save_predictions(predictions_dict, prediction_id)
        print("")
