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
    save_figs = True
    print_f1_score = True

    ckpt_folder_prefix = ''
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
    iou_threshold_report = 0.2

    # Plot settings
    title_fontsize = 9
    general_fontsize = 9
    marker_size = 7  # moda 5, others 7
    marker_alpha = 0.8  # moda 0.5, others 0.8
    show_subject_id = False
    group_by_subject = True
    subject_to_highlight = []  # inta [2, 7, 10], others []
    fold_monocolor = True
    show_fold_id = False
    show_grid = True
    show_mean = False
    show_quadrants = False
    weight_marker_by_density = False
    weight_marker_max_alpha = 1.0
    weight_marker_min_alpha = 0.2
    axis_markers = np.arange(0, 1 + 0.001, 0.1)
    minor_axis_markers = np.arange(0, 1 + 0.001, 0.1)

    metric_thr_to_print = 0.6

    # MODA specific filters:
    n_blocks_to_include = [10]
    phase_to_include = [1, 2]
    n_blocks_to_highlight = [2, 3, 10]
    phase_to_highlight = [1]

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
    # Highlight subjects for MODA:
    if dataset_name == constants.MODA_SS_NAME:
        for subject_id in dataset.all_ids:
            subject_n_blocks = dataset.data[subject_id]['n_blocks']
            subject_phase = dataset.data[subject_id]['phase']
            if (subject_n_blocks in n_blocks_to_highlight) and (subject_phase in phase_to_highlight):
                subject_to_highlight.append(subject_id)
    for grid_folder in grid_folder_list:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=viz.DPI)
        opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[os.path.join(ckpt_folder, grid_folder)]
        outputs = {'f1': [], 'miou': [], 'prec': [], 'rec': [], 'fold': [], 'subjects': [], 'densities': []}
        for k in selected_folds:
            # Retrieve relevant data
            eval_predictions = predictions_dict[grid_folder][k][evaluation_set]
            subject_ids = eval_predictions.get_ids()
            feed_d = FeederDataset(dataset, subject_ids, task_mode, which_expert=which_expert)
            events_list = feed_d.get_stamps()
            eval_predictions.set_probability_threshold(opt_thr_list[k])
            detections_list = eval_predictions.get_stamps()
            densities = []
            for subject_id in subject_ids:
                marks = feed_d.get_subject_stamps(subject_id)
                n_spindles = marks.shape[0]
                n2_pages = feed_d.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
                minutes_n2 = n2_pages.size * dataset.page_duration / 60
                density = n_spindles / minutes_n2
                densities.append(density)
            # Compute performance
            iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
            f1_score = metrics.metric_vs_iou_macro_average(
                events_list, detections_list, [iou_threshold_report],
                iou_matching_list=iou_matching_list, metric_name=constants.F1_SCORE, collapse_values=False)
            recall = metrics.metric_vs_iou_macro_average(
                events_list, detections_list, [iou_threshold_report],
                iou_matching_list=iou_matching_list, metric_name=constants.RECALL, collapse_values=False)
            precision = metrics.metric_vs_iou_macro_average(
                events_list, detections_list, [iou_threshold_report],
                iou_matching_list=iou_matching_list, metric_name=constants.PRECISION, collapse_values=False)
            nonzero_iou_list = [m[m > 0] for m in iou_matching_list]
            miou_list = []
            for nonzero_iou in nonzero_iou_list:
                if nonzero_iou.size > 0:
                    miou_list.append(np.mean(nonzero_iou))
                else:
                    miou_list.append(np.nan)
            outputs['f1'].append(f1_score[:, 0])
            outputs['prec'].append(precision[:, 0])
            outputs['rec'].append(recall[:, 0])
            outputs['miou'].append(miou_list)
            outputs['subjects'].append(subject_ids)
            outputs['fold'].append([k] * len(subject_ids))
            outputs['densities'].append(densities)
        if group_by_subject:
            subject_ids = np.unique(np.concatenate(outputs['subjects']))
            grouped_outputs = {
                s: {key: [] for key in outputs}
                for s in subject_ids}
            for k in selected_folds:
                for i, s in enumerate(outputs['subjects'][k]):
                    for key in outputs.keys():
                        grouped_outputs[s][key].append(outputs[key][k][i])
            outputs = {key: [] for key in grouped_outputs[subject_ids[0]].keys()}
            for s in subject_ids:
                for key in grouped_outputs[s].keys():
                    if key in ['fold', 'subjects']:
                        # Assign to the first fold in which it appears
                        outputs[key].append(grouped_outputs[s][key][0])
                    else:
                        outputs[key].append(np.nanmean(grouped_outputs[s][key]))
        else:
            for key in outputs.keys():
                outputs[key] = np.concatenate(outputs[key])
        sorted_loc = np.argsort(outputs['fold'])
        for key in outputs.keys():
            outputs[key] = np.asarray(outputs[key])[sorted_loc]
        outputs['densities'] = outputs['densities'] / np.max(outputs['densities'])
        # Filter subjects:
        if dataset_name == constants.MODA_SS_NAME:
            filt_outputs = {key: [] for key in outputs.keys()}
            for i in range(len(outputs['f1'])):
                subject_id = outputs['subjects'][i]
                subject_n_blocks = dataset.data[subject_id]['n_blocks']
                subject_phase = dataset.data[subject_id]['phase']
                if (subject_n_blocks in n_blocks_to_include) and (subject_phase in phase_to_include):
                    for key in outputs.keys():
                        filt_outputs[key].append(outputs[key][i])
            outputs = filt_outputs
        print("Subjects to plot:", len(outputs['f1']))
        if print_f1_score:
            print(grid_folder, "F1-score by subject")
            tmp_subject_ids = outputs['subjects']
            tmp_all_ids = np.unique(tmp_subject_ids)
            tmp_f1_score = outputs['f1']
            for s in tmp_all_ids:
                subject_locs = np.where(tmp_subject_ids == s)[0]
                subject_metrics = tmp_f1_score[subject_locs]
                print("    Subject %s:" % s, subject_metrics)
        # Plot
        folds_shown = []
        for i in range(len(outputs['f1'])):
            subject_id = outputs['subjects'][i]
            k = outputs['fold'][i]
            if fold_monocolor:
                color = viz.PALETTE['green'] if subject_id in subject_to_highlight else viz.PALETTE['blue']
                zorder = 10 if subject_id in subject_to_highlight else 5
            else:
                color = color_dict[k]
                zorder = 5
            label = 'Fold %d' % k if k not in folds_shown else None
            folds_shown.append(k)
            if weight_marker_by_density:
                alpha_range = weight_marker_max_alpha - weight_marker_min_alpha
                point_alpha = weight_marker_min_alpha + alpha_range * outputs['densities'][i]
            else:
                point_alpha = marker_alpha
            ax.plot(
                outputs['rec'][i], outputs['prec'][i],
                color=color, linestyle='None', alpha=point_alpha, zorder=zorder,
                markeredgewidth=0.0, marker='o', markersize=marker_size,
                label=label)
            if show_subject_id:
                if isinstance(subject_id, str):
                    single_id_to_show = int(subject_id[0] + subject_id[3:])
                    subject_id_fontsize = 3
                else:
                    single_id_to_show = subject_id
                    subject_id_fontsize = 4
                ax.annotate(
                    single_id_to_show, (outputs['rec'][i], outputs['prec'][i]),
                    horizontalalignment="center", verticalalignment="center",
                    fontsize=subject_id_fontsize, color="w", zorder=20)
            if outputs['rec'][i] <= metric_thr_to_print or outputs['prec'][i] <= metric_thr_to_print:
                print("Subject %s with Recall %1.4f and Precision %1.4f" % (
                    subject_id, outputs['rec'][i], outputs['prec'][i]))
        if show_mean:
            ax.plot(
                np.mean(outputs['rec']), np.mean(outputs['prec']),
                marker='o', markersize=marker_size / 2, linestyle="None",
                color=viz.GREY_COLORS[6])
        perf_str = plotter.get_performance_string(outputs)
        eval_str = "%s-%s" % ("SUBJECT", evaluation_set.upper())
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
            dens_str = 'weighted' if weight_marker_by_density else 'plain'
            fname = os.path.join(save_dir, "pr_bysubject_%s_%s" % (dens_str, grid_folder))
            plt.savefig('%s.png' % fname, dpi=viz.DPI, bbox_inches="tight", pad_inches=0.01)
            plt.savefig('%s.svg' % fname, dpi=viz.DPI, bbox_inches="tight", pad_inches=0.01)
            plt.savefig('%s.pdf' % fname, dpi=viz.DPI, bbox_inches="tight", pad_inches=0.01)
        else:
            plt.show()
        plt.close()
