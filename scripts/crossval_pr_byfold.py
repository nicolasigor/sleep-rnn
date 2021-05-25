from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    save_figs = False

    ckpt_folder_prefix = '20210520_thesis_ablation_5fold-cv_exp1'
    # You may specify certain runs within that ckpt_folder in grid_folder_list.
    # If None then all runs are returned
    grid_folder_list = None
    # You may specify certain folds within the experiment to plot.
    # If None, then all available folds are used
    selected_folds = None

    # Data settings
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    task_mode = constants.N2_RECORD
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True

    # Evaluation settings
    evaluation_set = constants.TEST_SUBSET
    average_mode = constants.MACRO_AVERAGE
    iou_threshold_report = 0.2

    # Plot settings
    title_fontsize = 9
    general_fontsize = 9
    marker_size = 7
    marker_alpha = 0.8  # 1.0, 0.6 if too crowded
    fold_monocolor = True
    show_fold_id = False
    show_grid = True
    show_mean = True
    show_quadrants = True
    axis_markers = np.arange(0, 1 + 0.001, 0.1)
    minor_axis_markers = np.arange(0, 1 + 0.001, 0.1)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    color_dict = plotter.get_fold_colors()
    dataset = reader.load_dataset(
        dataset_name, params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=False)
    ckpt_folder = '%s_%s_train_%s' % (ckpt_folder_prefix, task_mode, dataset_name)
    if grid_folder_list is None:
        experiment_path = os.path.join(RESULTS_PATH, 'predictions_%s' % dataset_name, ckpt_folder)
        grid_folder_list = os.listdir(experiment_path)
        grid_folder_list.sort()
    print('Grid settings to be used from %s:' % ckpt_folder)
    pprint(grid_folder_list)
    save_dir = os.path.join(RESULTS_PATH, 'auto_pr_figs', ckpt_folder)
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)
        print("Saving directory: %s" % save_dir)
    predictions_dict = {}
    for grid_folder in grid_folder_list:
        grid_folder_complete = os.path.join(ckpt_folder, grid_folder)
        predictions_dict[grid_folder] = reader.read_predictions_crossval(
            grid_folder_complete, dataset, task_mode)
    if selected_folds is None:
        selected_folds = list(predictions_dict[grid_folder_list[0]].keys())
        selected_folds.sort()
    print("Folds to plot: %s" % selected_folds)
    print("")
    result_id = '%s-%s-E%d-%s' % (
        dataset_name.split('_')[0].upper(),
        dataset_name.split('_')[1].upper(),
        which_expert,
        task_mode.upper())
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}
    for grid_folder in grid_folder_list:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=viz.DPI)
        opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[os.path.join(ckpt_folder, grid_folder)]
        outputs = {'f1': [], 'miou': [], 'prec': [], 'rec': []}
        for k in selected_folds:
            # Retrieve relevant data
            eval_predictions = predictions_dict[grid_folder][k][evaluation_set]
            subject_ids = eval_predictions.get_ids()
            feed_d = FeederDataset(dataset, subject_ids, task_mode, which_expert=which_expert)
            events_list = feed_d.get_stamps()
            eval_predictions.set_probability_threshold(opt_thr_list[k])
            detections_list = eval_predictions.get_stamps()
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
            outputs['f1'].append(f1_score[0])
            outputs['prec'].append(precision[0])
            outputs['rec'].append(recall[0])
            outputs['miou'].append(miou)
            # Plot
            color = viz.PALETTE['blue'] if fold_monocolor else color_dict[k]
            ax.plot(
                recall[0], precision[0],
                color=color, linestyle='None', alpha=marker_alpha,
                markeredgewidth=0.0, marker='o', markersize=marker_size,
                label='Fold %d' % k)
        if show_mean:
            ax.plot(
                np.mean(outputs['rec']), np.mean(outputs['prec']),
                marker='o', markersize=marker_size / 2, linestyle="None",
                color=viz.GREY_COLORS[6])
        perf_str = plotter.get_performance_string(outputs)
        eval_str = "%s-%s" % (average_mode.split("_")[0].upper(), evaluation_set.upper())
        ax.set_title(
            '%s\n%s, %s\n%s' % (
                grid_folder, eval_str, result_id, perf_str),
            fontsize=title_fontsize)
        ax.tick_params(labelsize=general_fontsize)
        ax.set_xlabel('Recall (IoU>%1.1f)' % iou_threshold_report, fontsize=general_fontsize)
        ax.set_ylabel('Precision (IoU>%1.1f)' % iou_threshold_report, fontsize=general_fontsize)
        if show_fold_id:
            ax.legend(loc='lower left', fontsize=general_fontsize)
        plotter.format_precision_recall_plot_simple(
            ax, axis_markers=axis_markers, show_quadrants=show_quadrants,
            show_grid=show_grid, minor_axis_markers=minor_axis_markers)
        plt.tight_layout()
        if save_figs:
            fname = os.path.join(save_dir, "pr_byfold_%s.png" % grid_folder)
            plt.savefig(fname, dpi=viz.DPI, bbox_inches="tight", pad_inches=0.01)
        else:
            plt.show()
        plt.close()
