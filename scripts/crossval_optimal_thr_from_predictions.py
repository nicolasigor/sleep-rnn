from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
from pprint import pprint

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleep.data.loader import load_dataset
from sleep.data import utils
from sleep.detection.feeder_dataset import FeederDataset
from sleep.detection import metrics
from sleep.common import constants
from sleep.common import checks
from sleep.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')
# SEED_LIST = [123, 234, 345, 456]
SEED_LIST = [123]

if __name__ == '__main__':

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir

    ckpt_folder = '20190506_bsf'
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_KC_NAME

    which_expert = 1
    verbose = False
    grid_folder_list = None
    # -----

    # Performance settings
    res_thr = 0.02
    start_thr = 0.3
    end_thr = 0.6

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    dataset = load_dataset(dataset_name)

    # Get training set ids
    print('Loading train set... ', end='', flush=True)
    all_train_ids = dataset.train_ids
    print('Done')

    full_ckpt_folder = '%s_%s_train_%s' \
                       % (ckpt_folder, task_mode, dataset_name)

    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            RESULTS_PATH,
            full_ckpt_folder
        ))
        print('Grid settings found:')
        pprint(grid_folder_list)

    print('')

    # Load predictions
    predictions_dict = {}
    n_seeds = len(SEED_LIST)

    for j, folder_name in enumerate(grid_folder_list):

        print('\nGrid setting: %s' % folder_name)
        predictions_dict[folder_name] = []
        for k in range(n_seeds):
            print('\n%d / %d' % (k+1, n_seeds))
            if j == 0:
                this_seed = SEED_LIST[k]
                print('Validation split seed: %d' % this_seed)
            # Restore predictions
            ckpt_path = os.path.abspath(os.path.join(
                RESULTS_PATH,
                'predictions_%s' % dataset_name,
                full_ckpt_folder,
                '%s' % folder_name,
                'seed%d' % k
            ))
            print('Loading predictions from %s' % ckpt_path)
            this_dict = {}
            filename = os.path.join(
                    ckpt_path,
                    'prediction_%s_%s.pkl' % (task_mode, constants.VAL_SUBSET))
            with open(filename, 'rb') as handle:
                this_pred = pickle.load(handle)
            predictions_dict[folder_name].append(this_pred)
    print('\nDone')

    # Adjust thr
    n_thr = int(np.round((end_thr - start_thr) / res_thr + 1))
    thr_list = [start_thr + res_thr * i for i in range(n_thr)]
    thr_list = np.asarray(thr_list)
    # print(thr_list)
    print('Number of thresholds to be evaluated: %d' % len(thr_list))

    # ---------------- Compute performance
    crossval_af1_mean = {}
    crossval_af1_std = {}
    for folder_name in grid_folder_list:
        print('\nGrid setting: %s' % folder_name)
        crossval_af1_mean[folder_name] = []
        crossval_af1_std[folder_name] = []
        for thr in thr_list:
            print('Processing thr %1.4f' % thr)
            val_af1 = []
            for k, seed in enumerate(SEED_LIST):
                # Validation split
                _, val_ids = utils.split_ids_list(
                    all_train_ids, seed=seed, verbose=verbose)
                if verbose:
                    print('Val IDs:', val_ids)
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
        print('Done')

    # Search optimum
    print('\nVal AF1 report for %s' % full_ckpt_folder)
    for j, folder_name in enumerate(grid_folder_list):
        max_idx = np.argmax(np.array(crossval_af1_mean[folder_name])).item()
        half_idx = np.where(np.isclose(thr_list, 0.5))[0].item()

        print('%1.4f +- %1.4f (mu 0.5), '
              '%1.4f +- %1.4f (mu %1.3f). for setting %s'
              % (crossval_af1_mean[folder_name][half_idx],
                 crossval_af1_std[folder_name][half_idx],
                 crossval_af1_mean[folder_name][max_idx],
                 crossval_af1_std[folder_name][max_idx],
                 thr_list[max_idx],
                 folder_name))
