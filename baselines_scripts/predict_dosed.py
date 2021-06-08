from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pprint import pprint
import sys
import pickle

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys

BASELINE_FOLDER = '2019_chambon_dosed'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))
BASELINES_SAVE_PATH = os.path.abspath(os.path.join(project_root, 'resources', 'comparison_data', 'baselines_2021'))


def get_fit_id(config):
    fit_id = "%s_e%d_%s" % (config["dataset_name"], config["which_expert"], config["strategy"])
    return fit_id


def get_predictions(config, pred_folder):
    pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'results', pred_folder)
    dataset = reader.load_dataset(config["dataset_name"], verbose=False, params={pkeys.FS: 200})
    _, _, test_ids_list = butils.get_partitions(dataset, config["strategy"], config["n_seeds"])
    n_folds = len(test_ids_list)
    pred_dict = butils.get_raw_dosed_marks(pred_dir, n_folds, dataset, print_thr=False)
    pred_dict = butils.postprocess_dosed_marks(dataset, pred_dict)
    return pred_dict


def save_predictions(pred_dict, filename):
    pred_save_dir = os.path.join(BASELINES_SAVE_PATH, BASELINE_FOLDER)
    with open(os.path.join(pred_save_dir, '%s.pkl' % filename), 'wb') as handle:
        pickle.dump(pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    isolated_configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1, event_path='spindle1',
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2, event_path='spindle2',
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1, event_path='kcomplex',
            strategy='fixed', n_seeds=11),
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1, event_path='kcomplex',
            strategy='5cv', n_seeds=3),
    ]
    cross_configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1, event_path='spindle1',
            strategy='5cv', n_seeds=3),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2, event_path='spindle2',
            strategy='5cv', n_seeds=3),
        dict(
            dataset_name=constants.MODA_SS_NAME, which_expert=1, event_path='spindle',
            strategy='5cv', n_seeds=3),
        dict(
            dataset_name=constants.INTA_SS_NAME, which_expert=1, event_path='spindle',
            strategy='5cv', n_seeds=3),
    ]

    # In-dataset
    for config in isolated_configs+cross_configs:
        prediction_id = "%s_from_%s" % (get_fit_id(config), get_fit_id(config))
        print("Predicting %s" % prediction_id)
        pred_folder = '20210605_thesis_%s_%s_%s_final' % (
            config["dataset_name"].split("_")[0], config["strategy"], config["event_path"])
        print('Loading predictions from %s' % pred_folder)
        predictions_dict = get_predictions(config, pred_folder)
        save_predictions(predictions_dict, prediction_id)
        print("")

    # Cross-dataset
    for source_config in cross_configs:
        for target_config in cross_configs:
            source_id = get_fit_id(source_config)
            target_id = get_fit_id(target_config)
            if source_id == target_id:
                continue
            prediction_id = "%s_from_%s" % (target_id, source_id)
            print("Predicting %s" % prediction_id)
            pred_folder = '20210607_thesis_%s_from_%s_%s_to_%s_%s_adapt' % (
                target_config["strategy"],
                source_config["dataset_name"].split("_")[0], source_config["event_path"],
                target_config["dataset_name"].split("_")[0], target_config["event_path"])
            print('Loading predictions from %s' % pred_folder)
            predictions_dict = get_predictions(target_config, pred_folder)
            save_predictions(predictions_dict, prediction_id)
            print("")
