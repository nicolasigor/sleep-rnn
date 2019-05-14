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

    ckpt_folder = '20190504_bsf'
    task_mode = constants.WN_RECORD
    dataset_name = constants.MASS_KC_NAME

    which_expert = 1
    verbose = False
    grid_folder_list = None
    # -----

    # Performance settings
    res_thr = 0.02
    start_thr = 0.3
    end_thr = 0.7

    set_list = [constants.TRAIN_SUBSET, constants.VAL_SUBSET]
    # set_list = [constants.VAL_SUBSET]

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
            this_pred_dict = {}
            for set_name in set_list:
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
                        'prediction_%s_%s.pkl' % (task_mode, set_name))
                with open(filename, 'rb') as handle:
                    this_pred_dict[set_name] = RefactorUnpickler(handle).load()
                print('Loaded seed %d/%d from %s' % (k + 1, n_seeds, ckpt_path))
            predictions_dict[folder_name].append(this_pred_dict)
    print('Done\n')

    # Adjust thr
    n_thr = int(np.round((end_thr - start_thr) / res_thr + 1))
    thr_list = np.array([start_thr + res_thr * i for i in range(n_thr)])
    thr_list = np.round(thr_list, 2)
    print('%d thresholds to be evaluated between %1.4f and %1.4f'
          % (n_thr, thr_list[0], thr_list[-1]))

    # ---------------- Compute performance
    # af1 is computed on alltrain set.
    per_seed_af1 = {}
    for folder_name in grid_folder_list:
        print('Evaluating grid setting: %s ' % folder_name, flush=True)
        per_seed_af1[folder_name] = {}
        for k, seed in enumerate(SEED_LIST):
            per_seed_af1[folder_name][k] = []
            # Validation split
            train_ids, val_ids = utils.split_ids_list(
                all_train_ids, seed=seed, verbose=verbose)
            ids_dict = {
                constants.TRAIN_SUBSET: train_ids,
                constants.VAL_SUBSET: val_ids}

            for thr in thr_list:
                events_list = []
                detections_list = []
                for set_name in set_list:
                    # Prepare expert labels
                    data_inference = FeederDataset(
                        dataset, ids_dict[set_name], task_mode,
                        which_expert=which_expert)
                    this_events = data_inference.get_stamps()
                    # Prepare model predictions
                    prediction_obj = predictions_dict[folder_name][k][set_name]
                    prediction_obj.set_probability_threshold(thr)
                    this_detections = prediction_obj.get_stamps()
                    events_list = events_list + this_events
                    detections_list = detections_list + this_detections
                # Compute AF1
                af1_at_thr = metrics.average_metric_with_list(
                    events_list, detections_list, verbose=False)
                per_seed_af1[folder_name][k].append(af1_at_thr)

    # Search optimum
    print('\nVal AF1 report for %s' % full_ckpt_folder)

    # half_idx = np.where(np.isclose(thr_list, 0.5))[0].item()
    # metric_to_sort = [
    #     - np.mean(
    #         [per_seed_af1[folder_name][k][half_idx] for k in range(n_seeds)]
    #     )
    #     for folder_name in grid_folder_list]
    # idx_sorted = np.argsort(metric_to_sort)
    # grid_folder_list = [grid_folder_list[k] for k in idx_sorted]

    for j, folder_name in enumerate(grid_folder_list):
        seeds_half_performance = []
        seeds_best_performance = []
        seeds_best_thr = []
        for k in range(n_seeds):
            this_af1 = per_seed_af1[folder_name][k]
            max_idx = np.argmax(this_af1).item()
            this_best_thr = thr_list[max_idx]
            seeds_best_thr.append(this_best_thr)

            # Now load validation set and compute performance
            _, val_ids = utils.split_ids_list(
                all_train_ids, seed=SEED_LIST[k], verbose=False)
            # Prepare expert labels
            data_inference = FeederDataset(
                dataset, val_ids, task_mode,
                which_expert=which_expert)
            this_events = data_inference.get_stamps()
            # Prepare model predictions
            prediction_obj = predictions_dict[folder_name][k][constants.VAL_SUBSET]

            # Half thr
            prediction_obj.set_probability_threshold(0.5)
            this_detections = prediction_obj.get_stamps()
            af1_at_thr = metrics.average_metric_with_list(
                this_events, this_detections, verbose=False)
            seeds_half_performance.append(af1_at_thr)

            # Best thr
            prediction_obj.set_probability_threshold(this_best_thr)
            this_detections = prediction_obj.get_stamps()
            af1_at_thr = metrics.average_metric_with_list(
                this_events, this_detections, verbose=False)
            seeds_best_performance.append(af1_at_thr)

            # seeds_best_performance.append(this_af1[max_idx])
            # seeds_half_performance.append(this_af1[half_idx])
        mean_best_performance = np.mean(seeds_best_performance).item()
        std_best_performance = np.std(seeds_best_performance).item()
        mean_half_performance = np.mean(seeds_half_performance).item()
        std_half_performance = np.std(seeds_half_performance).item()
        print('%1.4f +- %1.4f (mu 0.5), '
              '%1.4f +- %1.4f (mu %s) for setting %s'
              % (mean_half_performance,
                 std_half_performance,
                 mean_best_performance,
                 std_best_performance,
                 seeds_best_thr,
                 folder_name))