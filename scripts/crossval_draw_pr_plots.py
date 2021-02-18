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

from sleeprnn.helpers import reader, misc
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT
from sleeprnn.common import constants, pkeys, viz

RESULTS_PATH = os.path.join(project_root, 'results')

if __name__ == '__main__':

    ckpt_folder = ''
    # You may specify certain runs within that ckpt_folder in grid_folder_list.
    # If None then all runs are returned
    grid_folder_list = None

    dataset_name = constants.CAP_ALL_SS_NAME
    train_fraction = 0.86
    fs = 200
    which_expert = 1
    task_mode = constants.N2_RECORD
    set_list = [constants.VAL_SUBSET]
    verbose = False

    # Plot settings
    save_figs = True
    show_subject_id = True
    show_grid = True
    show_desired_attractor = True
    show_mean = True
    show_quadrants = True
    skip_subjects = []  # [11, 14, 19]
    iou_to_show = 0.2

    # ----------------------

    # Identifier
    result_id = '%s-%s-E%d-%s' % (
        dataset_name.split('_')[0].upper(),
        dataset_name.split('_')[1].upper(),
        which_expert,
        task_mode.upper())

    # Define paths
    full_ckpt_folder = '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name)
    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            RESULTS_PATH, 'predictions_%s' % dataset_name, full_ckpt_folder))
    grid_folder_list.sort()
    print("Checkpoint folder:\n%s" % full_ckpt_folder)
    print("Considered settings:")
    pprint(grid_folder_list)
    save_dir = os.path.join(
        RESULTS_PATH,
        'auto_pr_figs',
        full_ckpt_folder)
    # Find available seeds
    available_seed_folders = os.listdir(os.path.abspath(os.path.join(
        RESULTS_PATH,
        'predictions_%s' % dataset_name,
        full_ckpt_folder, grid_folder_list[0]
    )))
    seeds_to_show = [int(f[4:]) for f in available_seed_folders]
    seeds_to_show.sort()
    print("Available seeds: %s" % seeds_to_show)
    print("")

    # Load data and predictions
    dataset = reader.load_dataset(dataset_name, params={pkeys.FS: fs}, verbose=verbose)
    ids_dict = {constants.ALL_TRAIN_SUBSET: dataset.train_ids}
    ids_dict.update(misc.get_splits_dict(dataset, seeds_to_show, use_test_set=False, train_fraction=train_fraction))
    predictions_dict = {}
    for grid_folder in grid_folder_list:
        full_grid_path = os.path.join(full_ckpt_folder, grid_folder)
        predictions_dict[grid_folder] = reader.read_prediction_with_seeds(
            full_grid_path, dataset_name, task_mode, seeds_to_show, set_list=set_list, parent_dataset=dataset,
            verbose=verbose
        )

    # Plot
    print("\nProcessing %s" % full_ckpt_folder, flush=True)
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)
        print("Saving directory: %s" % save_dir)
    color_dict = {
        constants.TRAIN_SUBSET: {
            i: viz.GREY_COLORS[4] for i in range(4)},
        constants.VAL_SUBSET: {
            0: viz.PALETTE[constants.RED],
            1: viz.PALETTE[constants.BLUE],
            2: viz.PALETTE[constants.GREEN],
            3: viz.PALETTE[constants.DARK]}
    }
    markersize_model = 6
    axis_markers = np.arange(0, 1.1, 0.1)
    for grid_folder in grid_folder_list:
        full_grid_path = os.path.join(full_ckpt_folder, grid_folder)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=viz.DPI)
        tmp_all_recall = []
        tmp_all_precision = []
        tmp_all_mean_iou = []
        for seed_id in seeds_to_show:
            # ---------------- Compute performance
            pre_vs_iou_subject_dict = {}
            rec_vs_iou_subject_dict = {}
            mean_iou_subject_dict = {}
            for set_name in set_list:
                # Prepare expert labels
                data_inference = FeederDataset(dataset, ids_dict[seed_id][set_name], task_mode, which_expert)
                this_ids = data_inference.get_ids()
                this_events_list = data_inference.get_stamps()
                # Prepare model predictions
                prediction_obj = predictions_dict[grid_folder][seed_id][set_name]
                prediction_obj.set_probability_threshold(OPTIMAL_THR_FOR_CKPT_DICT[full_grid_path][seed_id])
                this_detections_list = prediction_obj.get_stamps()
                for i, single_id in enumerate(this_ids):
                    single_events = this_events_list[i]
                    single_detections = this_detections_list[i]
                    this_iou_matching, _ = metrics.matching(single_events, single_detections)
                    this_mean_iou = np.mean(this_iou_matching[this_iou_matching > 0])
                    this_precision = metrics.metric_vs_iou(
                        single_events, single_detections, [iou_to_show], metric_name=constants.PRECISION,
                        iou_matching=this_iou_matching)
                    this_recall = metrics.metric_vs_iou(
                        single_events, single_detections, [iou_to_show], metric_name=constants.RECALL,
                        iou_matching=this_iou_matching)
                    pre_vs_iou_subject_dict[single_id] = this_precision[0]
                    rec_vs_iou_subject_dict[single_id] = this_recall[0]
                    mean_iou_subject_dict[single_id] = this_mean_iou

            # -------------------- P L O T ----------------------
            for set_name in set_list:
                for i, single_id in enumerate(ids_dict[seed_id][set_name]):
                    if single_id in skip_subjects:
                        continue
                    this_rec = rec_vs_iou_subject_dict[single_id]
                    this_pre = pre_vs_iou_subject_dict[single_id]
                    this_iou = mean_iou_subject_dict[single_id]
                    tmp_all_recall.append(this_rec)
                    tmp_all_precision.append(this_pre)
                    tmp_all_mean_iou.append(this_iou)
                    label = 'Split %d' % seed_id if i == 0 else None
                    ax.plot(
                        this_rec, this_pre, color=color_dict[set_name][seed_id],
                        marker='o', markersize=markersize_model, label=label, linestyle='None')
                    if show_subject_id:
                        if isinstance(single_id, str):
                            single_id_to_show = int(single_id[2:])
                        else:
                            single_id_to_show = single_id
                        ax.annotate(
                            single_id_to_show, (this_rec, this_pre),
                            horizontalalignment="center", verticalalignment="center", fontsize=4, color="w")
        tmp_all_precision = np.array(tmp_all_precision)
        tmp_all_recall = np.array(tmp_all_recall)
        tmp_all_f1_score = 2 * tmp_all_precision * tmp_all_recall / (tmp_all_precision + tmp_all_recall)
        print("Precision %1.1f \u00B1 %4.1f -- Recall %1.1f \u00B1 %4.1f -- F1-score %1.1f \u00B1 %1.1f -- mIoU %1.1f \u00B1 %1.1f for %s" % (
            100 * np.mean(tmp_all_precision), 100 * np.std(tmp_all_precision),
            100 * np.mean(tmp_all_recall), 100 * np.std(tmp_all_recall),
            100 * np.mean(tmp_all_f1_score), 100 * np.std(tmp_all_f1_score),
            100 * np.mean(tmp_all_mean_iou), 100 * np.std(tmp_all_mean_iou),
            grid_folder
        ))
        perf_str = "F1: %1.1f\u00B1%1.1f, IoU: %1.1f\u00B1%1.1f\nP: %1.1f\u00B1%1.1f, R: %1.1f\u00B1%1.1f" % (
            100 * np.mean(tmp_all_f1_score), 100 * np.std(tmp_all_f1_score),
            100 * np.mean(tmp_all_mean_iou), 100 * np.std(tmp_all_mean_iou),
            100 * np.mean(tmp_all_precision), 100 * np.std(tmp_all_precision),
            100 * np.mean(tmp_all_recall), 100 * np.std(tmp_all_recall),
        )
        ax.plot([0, 1], [0, 1], zorder=1, linewidth=1, color=viz.GREY_COLORS[4])
        ax.set_title('%s\nValidation, %s\n%s' % (grid_folder, result_id, perf_str), fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_yticks(axis_markers)
        ax.set_xticks(axis_markers)
        if show_grid:
            ax.set_xticks(np.arange(0, 1, 0.1), minor=True)
            ax.set_yticks(np.arange(0, 1, 0.1), minor=True)
            ax.grid(which="minor")
        if show_quadrants:
            ax.axhline(0.5, color=viz.GREY_COLORS[5], linewidth=2)
            ax.axvline(0.5, color=viz.GREY_COLORS[5], linewidth=2)
        if show_desired_attractor:
            ax.fill_between([0.80, 0.9], 0.8, 0.9, facecolor=viz.GREY_COLORS[2], zorder=1)
        if show_mean:
            ax.plot(
                np.mean(tmp_all_recall), np.mean(tmp_all_precision),
                marker='o', markersize=markersize_model / 2, linestyle="None",
                color=viz.GREY_COLORS[6]
            )
        ax.tick_params(labelsize=8.5)
        ax.set_ylabel('Precision (IoU>%1.1f)' % iou_to_show, fontsize=9)
        ax.set_xlabel('Recall (IoU>%1.1f)' % iou_to_show, fontsize=9)
        ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        if save_figs:
            fname = os.path.join(save_dir, "pr_seeds_%s.png" % grid_folder)
            plt.savefig(fname, dpi=viz.DPI, bbox_inches="tight", pad_inches=0.01)
        plt.close()
