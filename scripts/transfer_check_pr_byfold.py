from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader, plotter
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT
from sleeprnn.common import constants, pkeys, viz

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == "__main__":
    save_figs = True

    target_dataset_config = (constants.MODA_SS_NAME, 1)
    source_dataset_configs = [
        (constants.MASS_SS_NAME, 1),
        (constants.MASS_SS_NAME, 2),
        (constants.INTA_SS_NAME, 1),
    ]
    ckpt_folder_prefix = '20210529_thesis_indata_5cv'
    transfer_date = '20210605'
    transfer_desc_list = ['sourcestd']

    # Evaluation settings
    evaluation_set = constants.TEST_SUBSET
    average_mode = constants.MICRO_AVERAGE
    iou_threshold_report = 0.2

    # Data settings
    task_mode = constants.N2_RECORD
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True

    # Plot settings
    marker_size = 5
    marker_alpha = 0.7

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # Load data
    dataset_name = target_dataset_config[0]
    which_expert = target_dataset_config[1]
    dataset = reader.load_dataset(
        dataset_name, params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=False)
    # Prepare paths
    ckpt_folder = '%s_e%d_%s_train_%s' % (ckpt_folder_prefix, which_expert, task_mode, dataset_name)
    indata_id = 'from %s-e%d-sourcestd' % (dataset_name, which_expert)
    transfer_ckpt_folders = {indata_id: ckpt_folder}
    for source_dataset_config, transfer_desc in itertools.product(source_dataset_configs, transfer_desc_list):
        transfer_ckpt_folder = '%s_from_%s_desc_%s_to_%s' % (
            transfer_date,
            '%s_e%d_%s_train_%s' % (
                ckpt_folder_prefix, source_dataset_config[1], task_mode, source_dataset_config[0]),
            transfer_desc,
            'e%d_%s_train_%s' % (which_expert, task_mode, dataset_name)
        )
        transfer_id = 'from %s-e%d-%s' % (source_dataset_config[0], source_dataset_config[1], transfer_desc)
        transfer_ckpt_folders[transfer_id] = transfer_ckpt_folder
    print("Prediction folders:")
    pprint(transfer_ckpt_folders)
    # Load grids from indata folder
    experiment_path = os.path.join(RESULTS_PATH, 'predictions_%s' % dataset_name, ckpt_folder)
    grid_folder_list = os.listdir(experiment_path)
    grid_folder_list.sort()
    print('Grid settings to be used from %s:' % ckpt_folder)
    pprint(grid_folder_list)
    # Load predictions
    predictions_dict = {}
    for grid_folder in grid_folder_list:
        predictions_dict[grid_folder] = {}
        for source_id in transfer_ckpt_folders.keys():
            source_ckpt_folder = transfer_ckpt_folders[source_id]
            grid_folder_complete = os.path.join(source_ckpt_folder, grid_folder)
            print("Loading predictions from %s" % grid_folder_complete)
            predictions_dict[grid_folder][source_id] = reader.read_predictions_crossval(
                grid_folder_complete, dataset, task_mode)
    # Retrieve folds
    selected_folds = list(predictions_dict[grid_folder_list[0]][indata_id].keys())
    selected_folds.sort()
    print("Folds to plot: %s" % selected_folds)
    print("")
    # Collect performance
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}
    results = {}
    for grid_folder in grid_folder_list:
        results[grid_folder] = {}
        for source_id in transfer_ckpt_folders.keys():
            results[grid_folder][source_id] = {'f1': [], 'miou': [], 'prec': [], 'rec': []}
            opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[os.path.join(
                transfer_ckpt_folders[source_id], grid_folder)]
            check_min_duration = 20
            check_max_duration = 0
            for k in selected_folds:
                # Retrieve relevant data
                eval_predictions = predictions_dict[grid_folder][source_id][k][evaluation_set]
                subject_ids = eval_predictions.get_ids()
                feed_d = FeederDataset(dataset, subject_ids, task_mode, which_expert=which_expert)
                events_list = feed_d.get_stamps()
                eval_predictions.set_probability_threshold(opt_thr_list[k])
                detections_list = eval_predictions.get_stamps()
                # Sanitiy check: durations
                all_detections = np.concatenate(detections_list, axis=0)
                all_durations = (all_detections[:, 1] - all_detections[:, 0] + 1) / dataset.fs
                check_min_duration = min(check_min_duration, all_durations.min())
                check_max_duration = max(check_max_duration, all_durations.max())
                # Compute performance
                iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
                f1_score = metric_vs_iou_fn_dict[average_mode](
                    events_list, detections_list, [iou_threshold_report],
                    iou_matching_list=iou_matching_list, metric_name=constants.F1_SCORE)
                recall = metric_vs_iou_fn_dict[average_mode](
                    events_list, detections_list, [iou_threshold_report],
                    iou_matching_list=iou_matching_list, metric_name=constants.RECALL)
                precision = metric_vs_iou_fn_dict[average_mode](
                    events_list, detections_list, [iou_threshold_report],
                    iou_matching_list=iou_matching_list, metric_name=constants.PRECISION)
                nonzero_iou_list = [iou_matching[iou_matching > 0] for iou_matching in iou_matching_list]
                if average_mode == constants.MACRO_AVERAGE:
                    miou_list = [np.mean(nonzero_iou) for nonzero_iou in nonzero_iou_list]
                    miou = np.mean(miou_list)
                elif average_mode == constants.MICRO_AVERAGE:
                    miou = np.concatenate(nonzero_iou_list).mean()
                else:
                    raise ValueError("Average mode %s invalid" % average_mode)
                results[grid_folder][source_id]['f1'].append(f1_score[0])
                results[grid_folder][source_id]['prec'].append(precision[0])
                results[grid_folder][source_id]['rec'].append(recall[0])
                results[grid_folder][source_id]['miou'].append(miou)
            # Sanity check reuslt
            print("Duration range for grid %s and source %s: %1.2f - %1.2f [s]" % (
                grid_folder, source_id, check_min_duration, check_max_duration))
    # Plot results, one plot per grid_folder
    for grid_folder in grid_folder_list:
        print("\nProcessing plot and statistics of grid: %s" % grid_folder)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=viz.DPI)
        for source_id in transfer_ckpt_folders.keys():
            # Draw scatter
            ax.plot(
                results[grid_folder][source_id]['rec'],
                results[grid_folder][source_id]['prec'],
                linestyle='None', alpha=marker_alpha,
                markeredgewidth=0.0, marker='o', markersize=marker_size,
                label=source_id)
            # Print mean +- std statistics
            print("%s: F1 %1.1f +- %1.1f, R %1.1f +- %1.1f, P %1.1f +- %1.1f, mIoU %1.1f+-%1.1f" % (
                source_id,
                100 * np.mean(results[grid_folder][source_id]['f1']),
                100 * np.std(results[grid_folder][source_id]['f1']),
                100 * np.mean(results[grid_folder][source_id]['rec']),
                100 * np.std(results[grid_folder][source_id]['rec']),
                100 * np.mean(results[grid_folder][source_id]['prec']),
                100 * np.std(results[grid_folder][source_id]['prec']),
                100 * np.mean(results[grid_folder][source_id]['miou']),
                100 * np.std(results[grid_folder][source_id]['miou']),
            ))
        ax.legend(fontsize=6, loc="lower left")
        ax.set_title(
            "Target %s-e%d (by-fold). Grid: %s" % (dataset_name, which_expert, grid_folder),
            fontsize=8, loc="left")
        plotter.format_precision_recall_plot_simple(
            ax, axis_range=(0.0, 1.0), show_quadrants=False, show_grid=True)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Recall (IoU>%1.1f)" % iou_threshold_report, fontsize=8)
        ax.set_ylabel("Precision (IoU>%1.1f)" % iou_threshold_report, fontsize=8)
        plt.tight_layout()
        if save_figs:
            fname = os.path.join("transfer_pr_%s-e%d_%s" % (dataset_name, which_expert, grid_folder))
            plt.savefig('%s.png' % fname, dpi=viz.DPI, bbox_inches="tight", pad_inches=0.01)
        else:
            plt.show()
