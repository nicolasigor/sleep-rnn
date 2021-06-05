from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pprint import pprint
import sys

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys

BASELINE_FOLDER = '2019_chambon_dosed'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))


if __name__ == '__main__':
    fs = 200
    iou_threshold_report = 0.2
    # Dataset training settings
    configs = [
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1,
            result_folder="20210603_thesis_mass_fixed_kcomplex",
            strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1,
            result_folder="20210604_thesis_mass_fixed_kcomplex_globalstd",
            strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
        # dict(
        #     dataset_name=constants.MASS_KC_NAME, which_expert=1,
        #     result_folder="20210603_thesis_mass_fixed_kcomplex",
        #     strategy='fixed', n_seeds=11, average_mode=constants.MACRO_AVERAGE),
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
        pred_dict = butils.get_raw_dosed_marks(pred_dir, n_folds, dataset, print_thr=True)
        pred_dict = butils.postprocess_dosed_marks(dataset, pred_dict)
        results = butils.evaluate_dosed_by_fold(
            dataset, which_expert, test_ids_list, pred_dict, average_mode, iou_threshold_report)
        print("\nTest performance (%s, iou >= %1.2f, from train to test) for %s" % (
            average_mode, iou_threshold_report, fit_id))
        butils.print_performance(results)
        del dataset
