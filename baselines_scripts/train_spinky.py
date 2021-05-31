from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys

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


if __name__ == '__main__':

    save_fitting = True

    # Dataset training settings
    fs = 200
    dataset_name = constants.MASS_KC_NAME
    which_expert = 1
    strategy = '5cv'  # {'fixed' or '5cv'}
    n_seeds = 3  # {11, 3}
    average_mode = constants.MACRO_AVERAGE
    iou_threshold_report = 0.2

    # ------------------------------------------------------

    # Load dataset
    dataset = reader.load_dataset(dataset_name, verbose=True, params={pkeys.FS: fs})
    # Predictions
    pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'output_thesis_%s' % dataset_name.split("_")[0])
    print('Loading predictions from %s' % pred_dir, flush=True)
    settings = butils.get_settings(pred_dir, extract_setting)
    pred_dict = butils.get_prediction_dict(dataset, pred_dir, settings, get_raw_marks)
    train_ids_list, _, test_ids_list = butils.get_partitions(dataset, strategy, n_seeds)
    fitted_setting_dict = butils.train_grid(dataset, which_expert, train_ids_list, pred_dict, average_mode)
    fit_id = "%s_e%d_%s" % (dataset_name, which_expert, strategy)

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
