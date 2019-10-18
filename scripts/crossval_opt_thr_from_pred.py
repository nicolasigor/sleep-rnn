from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.helpers.reader import RefactorUnpickler
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.detection.threshold_optimization import get_optimal_threshold
from sleeprnn.common import constants, pkeys

RESULTS_PATH = os.path.join(project_root, 'results')
SEED_LIST = [123, 234, 345, 456]

if __name__ == '__main__':

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir

    ckpt_folder = '20191017_elastic_grid_pte2'
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True

    new_split_version = True  # True from 20190620
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    id_try_list = [0, 1, 2, 3]

    which_expert = 1
    verbose = False
    grid_folder_list = None
    # -----

    # Performance settings
    res_thr = 0.02
    start_thr = 0.2
    end_thr = 0.8

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    dataset = load_dataset(
        dataset_name,
        params=dataset_params, load_checkpoint=load_dataset_from_ckpt)
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
    for j, folder_name in enumerate(grid_folder_list):
        predictions_dict[folder_name] = {}
        for k in id_try_list:
            this_pred_dict = {}
            # Restore predictions
            ckpt_path = os.path.abspath(os.path.join(
                RESULTS_PATH,
                'predictions_%s' % dataset_name,
                full_ckpt_folder,
                '%s' % folder_name,
                'seed%d' % k
            ))
            for set_name in set_list:
                this_dict = {}
                filename = os.path.join(
                        ckpt_path,
                        'prediction_%s_%s.pkl' % (task_mode, set_name))
                with open(filename, 'rb') as handle:
                    this_pred_dict[set_name] = RefactorUnpickler(handle).load()
            print('Loaded seed %d from %s' % (k, ckpt_path))
            predictions_dict[folder_name][k] = this_pred_dict
    print('Done\n')

    # ---------------- Compute performance
    # af1 is computed on alltrain set.
    per_seed_thr = {}
    for folder_name in grid_folder_list:
        print('Evaluating grid setting: %s ' % folder_name, flush=True)
        per_seed_thr[folder_name] = {}
        for k in id_try_list:
            print('Seed %d' % k)
            # Split to form validation set
            if new_split_version:
                train_ids, val_ids = utils.split_ids_list_v2(
                    all_train_ids, split_id=k)
            else:
                train_ids, val_ids = utils.split_ids_list(
                    all_train_ids, seed=SEED_LIST[k])
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
                end_thr=end_thr,
                verbose=verbose
            )
            per_seed_thr[folder_name][k] = best_thr

    # Search optimum
    print('\nVal AF1 report for %s' % full_ckpt_folder)

    metric_to_sort_list = []
    str_to_show_list = []
    for j, folder_name in enumerate(grid_folder_list):
        seeds_best_f1_at_iou = []
        seeds_half_performance = []
        seeds_best_performance = []
        seeds_best_thr = []
        seeds_ap_std = []
        seeds_ar_std = []
        for k in id_try_list:
            this_best_thr = per_seed_thr[folder_name][k]
            seeds_best_thr.append(this_best_thr)

            # Now load validation set and compute performance
            if new_split_version:
                train_ids, val_ids = utils.split_ids_list_v2(
                    all_train_ids, split_id=k, verbose=False)
            else:
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
            f1_at_thr = metrics.metric_vs_iou_with_list(
                this_events_val, this_detections_val, [0.2, 0.3])
            seeds_best_f1_at_iou.append(f1_at_thr)

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

        mean_f1_at_iou = np.stack(seeds_best_f1_at_iou, axis=0).mean(axis=0)
        std_f1_at_iou = np.stack(seeds_best_f1_at_iou, axis=0).std(axis=0)

        seeds_best_thr_string = ', '.join([
            '%1.2f' % single_thr for single_thr in seeds_best_thr])
        str_to_show = (
                'AF1 %1.2f/%1.2f [0.5] '
                '%1.2f/%1.2f [%s], '
                'F1(03) %1.2f/%1.2f, '
                'AP/AR-STD %1.2f/%1.2f for %s'
                % (100*mean_half_performance, 100*std_half_performance,
                   100*mean_best_performance, 100*std_best_performance,
                   seeds_best_thr_string,
                   100 * mean_f1_at_iou[1], 100 * std_f1_at_iou[1],
                   100*mean_std_ap, 100*mean_std_ar,
                   folder_name
                   ))

        metric_to_sort_list.append(mean_best_performance)
        str_to_show_list.append(str_to_show)

    # Sort by descending order
    idx_sorted = np.argsort(-np.asarray(metric_to_sort_list))
    str_to_show_list = [str_to_show_list[i] for i in idx_sorted]
    for str_to_show in str_to_show_list:
        print(str_to_show)
