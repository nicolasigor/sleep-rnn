from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from pprint import pprint
import time

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset, RefactorUnpickler
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.common import constants

RESULTS_PATH = os.path.join(project_root, 'results')
SEED_LIST = [123, 234, 345, 456]
# SEED_LIST = [123]


if __name__ == '__main__':

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir

    ckpt_folder = '20190516_bsf'
    task_mode = constants.N2_RECORD
    dataset_name = constants.INTA_SS_NAME

    which_expert = 1
    verbose = False
    grid_folder_list = None
    # -----

    # Performance settings
    res_thr = 0.02
    start_thr = 0.3
    end_thr = 0.8

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    dataset = load_dataset(dataset_name)
    # Get training set ids
    all_train_ids = dataset.train_ids

    full_ckpt_folder = '%s_%s_train_%s' \
                       % (ckpt_folder, task_mode, dataset_name)
    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            RESULTS_PATH,
            'predictions_%s' % dataset_name,
            full_ckpt_folder
        ))
        print('Grid settings found inside %s:' % full_ckpt_folder)
        pprint(grid_folder_list)
    print('')

    # Load predictions
    print('Loading predictions')
    predictions_dict = {}
    n_seeds = len(SEED_LIST)
    for j, folder_name in enumerate(grid_folder_list):
        predictions_dict[folder_name] = []
        for k in range(n_seeds):
            # Restore predictions
            ckpt_path = os.path.abspath(os.path.join(
                RESULTS_PATH,
                'predictions_%s' % dataset_name,
                full_ckpt_folder,
                '%s' % folder_name,
                'seed%d' % k
            ))
            this_dict = {}
            filename = os.path.join(
                    ckpt_path,
                    'prediction_%s_%s.pkl' % (task_mode, constants.VAL_SUBSET))
            with open(filename, 'rb') as handle:
                this_pred = RefactorUnpickler(handle).load()
            predictions_dict[folder_name].append(this_pred)
            print('Loaded seed %d/%d from %s' % (k + 1, n_seeds, ckpt_path))
    print('Done\n')

    # Adjust thr
    n_thr = int(np.round((end_thr - start_thr) / res_thr + 1))
    thr_list = np.array([start_thr + res_thr * i for i in range(n_thr)])
    print('%d thresholds to be evaluated between %1.4f and %1.4f'
          % (n_thr, thr_list[0], thr_list[-1]))

    # ---------------- Compute performance
    crossval_af1_mean = {}
    crossval_af1_std = {}
    for folder_name in grid_folder_list:
        print('Evaluating grid setting: %s ' % folder_name, flush=True)
        crossval_af1_mean[folder_name] = []
        crossval_af1_std[folder_name] = []
        for thr in thr_list:
            val_af1 = []
            for k, seed in enumerate(SEED_LIST):
                # Validation split
                _, val_ids = utils.split_ids_list(
                    all_train_ids, seed=seed, verbose=verbose)
                # Prepare expert labels
                data_val = FeederDataset(
                    dataset, val_ids, task_mode, which_expert=which_expert)
                events_val = data_val.get_stamps()
                # Prepare model predictions
                prediction_val = predictions_dict[folder_name][k]
                prediction_val.set_probability_threshold(thr)
                detections_val = prediction_val.get_stamps()
                # Compute AF1
                val_af1_at_thr = metrics.average_metric_with_list(
                    events_val, detections_val, verbose=False)
                val_af1.append(val_af1_at_thr)
            crossval_af1_mean[folder_name].append(np.mean(val_af1))
            crossval_af1_std[folder_name].append(np.std(val_af1))

    # Search optimum
    print('\nVal AF1 report for %s' % full_ckpt_folder)

    half_idx = np.where(np.isclose(thr_list, 0.5))[0].item()
    metric_to_sort = [
        - crossval_af1_mean[folder_name][half_idx]
        for folder_name in grid_folder_list]
    idx_sorted = np.argsort(metric_to_sort)
    grid_folder_list = [grid_folder_list[k] for k in idx_sorted]

    for j, folder_name in enumerate(grid_folder_list):
        max_idx = np.argmax(np.array(crossval_af1_mean[folder_name])).item()
        print('%1.4f +- %1.4f (mu 0.5), '
              '%1.4f +- %1.4f (mu %1.3f) for setting %s'
              % (crossval_af1_mean[folder_name][half_idx],
                 crossval_af1_std[folder_name][half_idx],
                 crossval_af1_mean[folder_name][max_idx],
                 crossval_af1_std[folder_name][max_idx],
                 thr_list[max_idx],
                 folder_name))
