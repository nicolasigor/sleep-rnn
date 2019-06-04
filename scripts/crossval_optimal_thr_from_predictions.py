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
from sleeprnn.detection.threshold_optimization import get_optimal_threshold
from sleeprnn.common import constants

RESULTS_PATH = os.path.join(project_root, 'results')
SEED_LIST = [123, 234, 345, 456]
# SEED_LIST = [123]


if __name__ == '__main__':

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir

    ckpt_folder = '20190602_bsf_v14'
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME

    which_expert = 1
    verbose = False
    grid_folder_list = None
    # -----

    # Performance settings
    res_thr = 0.02
    start_thr = 0.2
    end_thr = 0.7

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    dataset = load_dataset(dataset_name)
    all_train_ids = dataset.train_ids

    full_ckpt_folder = '%s_%s_train_%s' \
                       % (ckpt_folder, task_mode, dataset_name)
    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            RESULTS_PATH,
            'predictions_%s' % dataset_name,
            full_ckpt_folder
        ))
        grid_folder_list.sort()
        print('Grid settings found inside %s:' % full_ckpt_folder)
        pprint(grid_folder_list)
    print('')

    set_list = [constants.TRAIN_SUBSET, constants.VAL_SUBSET]
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

    # ---------------- Compute performance
    # af1 is computed on alltrain set.
    per_seed_thr = {}
    for folder_name in grid_folder_list:
        print('Evaluating grid setting: %s ' % folder_name, flush=True)
        per_seed_thr[folder_name] = {}
        for k, seed in enumerate(SEED_LIST):
            print('Seed %d' % k)
            # Validation split
            train_ids, val_ids = utils.split_ids_list(
                all_train_ids, seed=seed, verbose=verbose)
            ids_dict = {
                constants.TRAIN_SUBSET: train_ids,
                constants.VAL_SUBSET: val_ids}

            feeder_dataset_list = []
            predicted_dataset_list = []
            for set_name in set_list:
                data_inference = FeederDataset(
                    dataset, ids_dict[set_name], task_mode,
                    which_expert=which_expert)
                feeder_dataset_list.append(data_inference)
                prediction_obj = predictions_dict[folder_name][k][set_name]
                predicted_dataset_list.append(prediction_obj)
            best_thr = get_optimal_threshold(
                feeder_dataset_list,
                predicted_dataset_list,
                res_thr=res_thr,
                start_thr=start_thr,
                end_thr=end_thr)
            per_seed_thr[folder_name][k] = best_thr

    # Search optimum
    print('\nVal AF1 report for %s' % full_ckpt_folder)

    for j, folder_name in enumerate(grid_folder_list):
        seeds_half_performance = []
        seeds_best_performance = []
        seeds_best_thr = []
        seeds_ap_std = []
        seeds_ar_std = []
        for k in range(n_seeds):
            this_best_thr = per_seed_thr[folder_name][k]
            seeds_best_thr.append(this_best_thr)

            # Now load validation set and compute performance
            train_ids, val_ids = utils.split_ids_list(
                all_train_ids, seed=SEED_LIST[k], verbose=False)
            # Prepare expert labels
            data_val = FeederDataset(
                dataset, val_ids, task_mode, which_expert=which_expert)
            data_train = FeederDataset(
                dataset, train_ids, task_mode, which_expert=which_expert)
            this_events_val = data_val.get_stamps()
            this_events_train = data_train.get_stamps()
            # Prepare model predictions
            prediction_obj_val = predictions_dict[folder_name][k][constants.VAL_SUBSET]
            prediction_obj_train = predictions_dict[folder_name][k][constants.TRAIN_SUBSET]

            # Half thr
            prediction_obj_val.set_probability_threshold(0.5)
            this_detections_val = prediction_obj_val.get_stamps()
            af1_at_thr = metrics.average_metric_with_list(
                this_events_val, this_detections_val, verbose=False)
            seeds_half_performance.append(af1_at_thr)

            # Best thr
            prediction_obj_val.set_probability_threshold(this_best_thr)
            prediction_obj_train.set_probability_threshold(this_best_thr)
            this_detections_val = prediction_obj_val.get_stamps()
            this_detections_train = prediction_obj_train.get_stamps()
            af1_at_thr = metrics.average_metric_with_list(
                this_events_val, this_detections_val, verbose=False)

            # Compute std for average recall and precision
            alltrain_events = this_events_val + this_events_train
            alltrain_detections = this_detections_val + this_detections_train
            ar_list = [
                metrics.average_metric(
                    single_events, single_detections,
                    verbose=False, metric_name=constants.RECALL
                )
                for (single_events, single_detections)
                in zip(alltrain_events, alltrain_detections)
            ]
            ap_list = [
                metrics.average_metric(
                    single_events, single_detections,
                    verbose=False, metric_name=constants.PRECISION
                )
                for (single_events, single_detections)
                in zip(alltrain_events, alltrain_detections)
            ]
            seeds_ap_std.append(np.std(ap_list))
            seeds_ar_std.append(np.std(ar_list))
            seeds_best_performance.append(af1_at_thr)

        mean_best_performance = np.mean(seeds_best_performance).item()
        std_best_performance = np.std(seeds_best_performance).item()
        mean_half_performance = np.mean(seeds_half_performance).item()
        std_half_performance = np.std(seeds_half_performance).item()
        mean_std_ap = np.mean(seeds_ap_std).item()
        mean_std_ar = np.mean(seeds_ar_std).item()
        print('AF1 %1.4f +- %1.4f (mu 0.5), '
              'AF1 %1.4f +- %1.4f (mu %s), '
              'AP-STD %1.4f AR-STD %1.4f '
              'for setting %s'
              % (mean_half_performance, std_half_performance,
                 mean_best_performance, std_best_performance, seeds_best_thr,
                 mean_std_ap, mean_std_ar,
                 folder_name))
