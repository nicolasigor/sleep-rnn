from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset, read_prediction_with_seeds
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

    ckpt_folder = '20210212_cap_long_training'
    id_try_list = [0, 1, 2, 3]
    dataset_name = constants.CAP_SS_NAME
    which_expert = 1

    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True
    task_mode = constants.N2_RECORD
    verbose = False
    grid_folder_list = None
    # -----

    # Performance settings
    res_thr = 0.02
    start_thr = 0.2
    end_thr = 0.8

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    print("Evaluation of optimal thresholds from predictions")
    print('Seeds: %s' % id_try_list)
    print("Thresholds: %1.2f:%1.2f:%1.2f" % (start_thr, res_thr, end_thr))

    dataset = load_dataset(
        dataset_name,
        params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=verbose)
    all_train_ids = dataset.train_ids
    print("Loaded: dataset %s at %s Hz" % (dataset.dataset_name, dataset.fs))

    full_ckpt_folder = '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name)
    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            RESULTS_PATH,
            'predictions_%s' % dataset_name,
            full_ckpt_folder
        ))
        grid_folder_list.sort()
    print('Grid settings to be evaluated from %s:' % full_ckpt_folder)
    pprint(grid_folder_list)

    # Load predictions
    set_list = [constants.TRAIN_SUBSET, constants.VAL_SUBSET]
    predictions_dict = {}
    id_try_list_for_preds = [k for k in id_try_list if k is not None]
    for folder_name in grid_folder_list:
        this_ckpt_folder = os.path.join(full_ckpt_folder, folder_name)
        predictions_dict[folder_name] = read_prediction_with_seeds(
            this_ckpt_folder, dataset_name, task_mode, id_try_list_for_preds,
            set_list=set_list, parent_dataset=dataset, verbose=verbose)
    print('Loaded: predictions.\n')

    # ---------------- Compute performance
    # af1 is computed on alltrain set to optimize threshold.
    per_seed_thr = {}
    for folder_name in grid_folder_list:
        print('Evaluating "%s"' % folder_name, flush=True)
        per_seed_thr[folder_name] = {}
        for k in id_try_list:
            if k is None:
                continue
            print('Seed %d' % k)
            # Split to form validation set
            train_ids, val_ids = utils.split_ids_list_v2(
                all_train_ids, split_id=k)
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
    str_to_register_list = []
    for j, folder_name in enumerate(grid_folder_list):
        seeds_best_f1_at_iou = []
        seeds_half_performance = []
        seeds_best_performance = []
        seeds_best_thr = {}
        seeds_val_precision = []
        seeds_val_recall = []

        for k in id_try_list:
            if k is None:
                continue
            this_best_thr = per_seed_thr[folder_name][k]
            seeds_best_thr[k] = this_best_thr

            # Now load validation set and compute performance
            _, val_ids = utils.split_ids_list_v2(
                all_train_ids, split_id=k, verbose=False)
            # Prepare expert labels
            data_val = FeederDataset(
                dataset, val_ids, task_mode, which_expert=which_expert)
            this_events_val = data_val.get_stamps()
            prediction_obj_val = predictions_dict[folder_name][k][constants.VAL_SUBSET]

            # Half thr
            prediction_obj_val.set_probability_threshold(0.5)
            this_detections_val = prediction_obj_val.get_stamps()
            af1_at_thr = metrics.average_metric_with_list(
                this_events_val, this_detections_val, verbose=False)
            seeds_half_performance.append(af1_at_thr)

            # Best thr
            prediction_obj_val.set_probability_threshold(this_best_thr)
            this_detections_val = prediction_obj_val.get_stamps()
            af1_at_thr = metrics.average_metric_with_list(
                this_events_val, this_detections_val, verbose=False)
            f1_at_thr = metrics.metric_vs_iou_with_list(
                this_events_val, this_detections_val, [0.2, 0.3])
            seeds_best_f1_at_iou.append(f1_at_thr)
            seeds_best_performance.append(af1_at_thr)

            # Compute dispersion
            t_precision_list = [
                metrics.metric_vs_iou(
                    single_events, single_detections,
                    [0.3], verbose=False,
                    metric_name=constants.PRECISION)
                for (single_events, single_detections)
                in zip(this_events_val, this_detections_val)
            ]
            t_recall_list = [
                metrics.metric_vs_iou(
                    single_events, single_detections,
                    [0.3], verbose=False,
                    metric_name=constants.RECALL)
                for (single_events, single_detections)
                in zip(this_events_val, this_detections_val)
            ]
            seeds_val_precision.append(t_precision_list)
            seeds_val_recall.append(t_recall_list)

        mean_best_performance = np.mean(seeds_best_performance).item()
        std_best_performance = np.std(seeds_best_performance).item()
        mean_half_performance = np.mean(seeds_half_performance).item()
        std_half_performance = np.std(seeds_half_performance).item()
        std_val_precision = np.array(seeds_val_precision).flatten().std().item()
        std_val_recall = np.array(seeds_val_recall).flatten().std().item()
        mean_val_precision = np.array(seeds_val_precision).flatten().mean().item()
        mean_val_recall = np.array(seeds_val_recall).flatten().mean().item()

        mean_f1_at_iou = np.stack(seeds_best_f1_at_iou, axis=0).mean(axis=0)
        std_f1_at_iou = np.stack(seeds_best_f1_at_iou, axis=0).std(axis=0)

        seeds_best_thr_string = ', '.join([
            'None' if k is None else '%1.2f' % seeds_best_thr[k] for k in id_try_list])
        str_to_show = (
                'AF1 %1.2f/%1.2f [0.5] '
                '%1.2f/%1.2f [%s], '
                'F3 %1.2f/%1.2f, '
                'P3 %1.1f/%s, '
                'R3 %1.1f/%s for %s'
                % (100*mean_half_performance, 100*std_half_performance,
                   100*mean_best_performance, 100*std_best_performance,
                   seeds_best_thr_string,
                   100 * mean_f1_at_iou[1], 100 * std_f1_at_iou[1],
                   100 * mean_val_precision,
                   ('%1.1f' % (100 * std_val_precision)).rjust(4),
                   100 * mean_val_recall,
                   ('%1.1f' % (100 * std_val_recall)).rjust(4),
                   folder_name
                   ))
        str_to_register = "    os.path.join('%s', '%s'): [%s]," % (
            full_ckpt_folder, folder_name, seeds_best_thr_string)

        metric_to_sort_list.append(mean_best_performance)
        str_to_show_list.append(str_to_show)
        str_to_register_list.append(str_to_register)

    # Sort by descending order
    idx_sorted = np.argsort(-np.asarray(metric_to_sort_list))
    str_to_show_list = [str_to_show_list[i] for i in idx_sorted]
    str_to_register_list = [str_to_register_list[i] for i in idx_sorted]
    for str_to_show in str_to_show_list:
        print(str_to_show)

    print("")
    for str_to_show in str_to_register_list:
        print(str_to_show)
