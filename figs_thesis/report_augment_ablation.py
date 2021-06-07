from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader, plotter
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT
from sleeprnn.common import constants, pkeys, viz

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == "__main__":
    ckpt_folder_prefix = '20210606_augment_ablation_6seeds_5cv_e1'
    grid_for_reference = "v2_time_wave0"

    # Data settings
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    task_mode = constants.N2_RECORD
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True

    # Evaluation settings
    evaluation_set = constants.TEST_SUBSET
    iou_threshold_report = 0.2

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    dataset = reader.load_dataset(
        dataset_name, params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=False)
    ckpt_folder = '%s_%s_train_%s' % (ckpt_folder_prefix, task_mode, dataset_name)
    experiment_path = os.path.join(RESULTS_PATH, 'predictions_%s' % dataset_name, ckpt_folder)
    grid_folder_list = os.listdir(experiment_path)
    grid_folder_list.sort()
    print('Grid settings to be used from %s:' % ckpt_folder)
    pprint(grid_folder_list)
    predictions_dict = {}
    for grid_folder in grid_folder_list:
        grid_folder_complete = os.path.join(ckpt_folder, grid_folder)
        predictions_dict[grid_folder] = reader.read_predictions_crossval(
            grid_folder_complete, dataset, task_mode)
    selected_folds = list(predictions_dict[grid_folder_list[0]].keys())
    selected_folds.sort()
    print("Folds to plot: %s" % selected_folds)
    print("")
    result_id = '%s-%s-E%d-%s' % (
        dataset_name.split('_')[0].upper(),
        dataset_name.split('_')[1].upper(),
        which_expert,
        task_mode.upper())
    data_comparison = {}
    for grid_folder in grid_folder_list:
        opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[os.path.join(ckpt_folder, grid_folder)]
        outputs = {'f1': [], 'rec': [], 'prec': [], 'miou': [], 'subjects': [], 'fold': []}
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
            outputs['rec'].append(recall[:, 0])
            outputs['prec'].append(precision[:, 0])
            outputs['miou'].append(miou_list)
            outputs['subjects'].append(subject_ids)
            outputs['fold'].append([k] * len(subject_ids))
        for key in outputs.keys():
            outputs[key] = np.concatenate(outputs[key])
        print("Setting:", grid_folder)

        outputs_df = pd.DataFrame.from_dict(outputs)
        data_comparison[grid_folder] = outputs_df
    # Latex ready print
    print("Latex ready:")
    metric_keys = ["f1", "rec", "prec", "miou"]
    print("Metrics: %s" % metric_keys)
    for grid_folder in grid_folder_list:
        bysubject_outputs = data_comparison[grid_folder].groupby(by="subjects").mean().drop(columns=["fold"])
        bysubject_stats = pd.concat({"mean": bysubject_outputs.mean(), "std": bysubject_outputs.std(ddof=0)}, axis=1)
        byfold_outputs = data_comparison[grid_folder].groupby(by="fold").mean().drop(columns=["subjects"])
        byfold_stats = pd.concat({"mean": byfold_outputs.mean(), "std": byfold_outputs.std(ddof=0)}, axis=1)
        msg = grid_folder
        for key in metric_keys:
            perf_str = " & $%1.1f\pm %1.1f\ (%1.1f)$" % (
                100 * byfold_stats.at[key, "mean"],
                100 * byfold_stats.at[key, "std"],
                100 * bysubject_stats.at[key, "std"])
            msg = msg + perf_str
        msg = msg + "\\\\"
        print(msg)
    # Now compute p-values
    print("\nStatistical tests\n")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        byfold_outputs = data_comparison[grid_folder].groupby(by="fold").mean().drop(columns=["subjects"])
        byfold_outputs_ref = data_comparison[grid_for_reference].groupby(by="fold").mean().drop(columns=["subjects"])
        f1_scores = byfold_outputs.f1.values
        f1_scores_ref = byfold_outputs_ref.f1.values
        pvalue = stats.ttest_ind(f1_scores_ref, f1_scores, equal_var=False)[1]
        print("F1-score (by-fold) %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
            grid_for_reference, f1_scores_ref.mean() * 100, f1_scores_ref.std() * 100,
            grid_folder, f1_scores.mean() * 100, f1_scores.std() * 100,
            pvalue))
    print("")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        bysubject_outputs = data_comparison[grid_folder].groupby(by="subjects").mean().drop(columns=["fold"])
        bysubject_outputs_ref = data_comparison[grid_for_reference].groupby(by="subjects").mean().drop(columns=["fold"])
        f1_scores = bysubject_outputs.f1.values
        f1_scores_ref = bysubject_outputs_ref.f1.values
        pvalue = stats.ttest_ind(f1_scores_ref, f1_scores, equal_var=False)[1]
        print("F1-score (by-subject) %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
            grid_for_reference, f1_scores_ref.mean() * 100, f1_scores_ref.std() * 100,
            grid_folder, f1_scores.mean() * 100, f1_scores.std() * 100,
            pvalue))
    print("")
    # Each subject individually
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        subject_ids = np.unique(data_comparison[grid_folder].subjects.values)
        for subject_id in subject_ids:
            df_curr = data_comparison[grid_folder]
            df_ref = data_comparison[grid_for_reference]
            subject_f1 = df_curr.loc[df_curr['subjects'] == subject_id].f1.values
            subject_f1_ref = df_ref.loc[df_ref['subjects'] == subject_id].f1.values
            pvalue = stats.ttest_ind(subject_f1_ref, subject_f1, equal_var=False)[1]
            print("    S%02d. F1-score %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
                subject_id,
                grid_for_reference, subject_f1_ref.mean() * 100, subject_f1_ref.std() * 100,
                grid_folder, subject_f1.mean() * 100, subject_f1.std() * 100,
                pvalue))
        print("")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        subject_ids = np.unique(data_comparison[grid_folder].subjects.values)
        for subject_id in subject_ids:
            df_curr = data_comparison[grid_folder]
            df_ref = data_comparison[grid_for_reference]
            subject_rec = df_curr.loc[df_curr['subjects'] == subject_id].rec.values
            subject_rec_ref = df_ref.loc[df_ref['subjects'] == subject_id].rec.values
            pvalue = stats.ttest_ind(subject_rec_ref, subject_rec, equal_var=False)[1]
            print("    S%02d. Recall %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
                subject_id,
                grid_for_reference, subject_rec_ref.mean() * 100, subject_rec_ref.std() * 100,
                grid_folder, subject_rec.mean() * 100, subject_rec.std() * 100,
                pvalue))
        print("")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        subject_ids = np.unique(data_comparison[grid_folder].subjects.values)
        for subject_id in subject_ids:
            df_curr = data_comparison[grid_folder]
            df_ref = data_comparison[grid_for_reference]
            subject_prec = df_curr.loc[df_curr['subjects'] == subject_id].prec.values
            subject_prec_ref = df_ref.loc[df_ref['subjects'] == subject_id].prec.values
            pvalue = stats.ttest_ind(subject_prec_ref, subject_prec, equal_var=False)[1]
            print("    S%02d. Precision %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
                subject_id,
                grid_for_reference, subject_prec_ref.mean() * 100, subject_prec_ref.std() * 100,
                grid_folder, subject_prec.mean() * 100, subject_prec.std() * 100,
                pvalue))
        print("")

    # Draw figure
    marker_size = 5
    marker_alpha = 0.7
    grid_colors = {
        "v2_time_wave0": viz.GREY_COLORS[6],
        "v2_time_wave1": viz.PALETTE["blue"]
    }
    plot_labels = {
        "v2_time_wave0": "Sin aumento",
        "v2_time_wave1": "Con aumento"
    }
    event_id = "E1SS" if dataset_name == constants.MASS_SS_NAME else "KC"
    # fig, axes = plt.subplots(1, 2, figsize=(4, 2.8), dpi=200, sharex=True, sharey=True)
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.8), dpi=200, sharex=True, sharey=True)
    ax = axes[0]
    for grid_folder in grid_folder_list:
        byfold_outputs = data_comparison[grid_folder].groupby(by="fold").mean()
        recall = byfold_outputs.rec.values
        precision = byfold_outputs.prec.values
        ax.plot(
            recall, precision, color=grid_colors[grid_folder], linestyle='None', alpha=marker_alpha,
            markeredgewidth=0.0, marker='o', markersize=marker_size, label=plot_labels[grid_folder])
        ax.set_ylabel('Precision (IoU>%1.1f)' % iou_threshold_report, fontsize=8)
        ax.set_title("Particiones (evento %s)" % event_id, fontsize=8, loc="left")
    ax = axes[1]
    for grid_folder in grid_folder_list:
        bysubject_outputs = data_comparison[grid_folder].groupby(by="subjects").mean()
        recall = bysubject_outputs.rec.values
        precision = bysubject_outputs.prec.values
        ax.plot(
            recall, precision, color=grid_colors[grid_folder], linestyle='None', alpha=marker_alpha,
            markeredgewidth=0.0, marker='o', markersize=marker_size, label=plot_labels[grid_folder])
        ax.set_title("Sujetos (evento %s)" % event_id, fontsize=8, loc="left")

    # ----
    # ----

    ckpt_folder_prefix = '20210606_augment_ablation_6seeds_5cv_e1'
    grid_for_reference = "v2_time_wave0"

    # Data settings
    dataset_name = constants.MASS_KC_NAME
    which_expert = 1
    task_mode = constants.N2_RECORD
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True

    # Evaluation settings
    evaluation_set = constants.TEST_SUBSET
    iou_threshold_report = 0.2

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    dataset = reader.load_dataset(
        dataset_name, params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=False)
    ckpt_folder = '%s_%s_train_%s' % (ckpt_folder_prefix, task_mode, dataset_name)
    experiment_path = os.path.join(RESULTS_PATH, 'predictions_%s' % dataset_name, ckpt_folder)
    grid_folder_list = os.listdir(experiment_path)
    grid_folder_list.sort()
    print('Grid settings to be used from %s:' % ckpt_folder)
    pprint(grid_folder_list)
    predictions_dict = {}
    for grid_folder in grid_folder_list:
        grid_folder_complete = os.path.join(ckpt_folder, grid_folder)
        predictions_dict[grid_folder] = reader.read_predictions_crossval(
            grid_folder_complete, dataset, task_mode)
    selected_folds = list(predictions_dict[grid_folder_list[0]].keys())
    selected_folds.sort()
    print("Folds to plot: %s" % selected_folds)
    print("")
    result_id = '%s-%s-E%d-%s' % (
        dataset_name.split('_')[0].upper(),
        dataset_name.split('_')[1].upper(),
        which_expert,
        task_mode.upper())
    data_comparison = {}
    for grid_folder in grid_folder_list:
        opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[os.path.join(ckpt_folder, grid_folder)]
        outputs = {'f1': [], 'rec': [], 'prec': [], 'miou': [], 'subjects': [], 'fold': []}
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
            outputs['rec'].append(recall[:, 0])
            outputs['prec'].append(precision[:, 0])
            outputs['miou'].append(miou_list)
            outputs['subjects'].append(subject_ids)
            outputs['fold'].append([k] * len(subject_ids))
        for key in outputs.keys():
            outputs[key] = np.concatenate(outputs[key])
        print("Setting:", grid_folder)

        outputs_df = pd.DataFrame.from_dict(outputs)
        data_comparison[grid_folder] = outputs_df
    # Latex ready print
    print("Latex ready:")
    metric_keys = ["f1", "rec", "prec", "miou"]
    print("Metrics: %s" % metric_keys)
    for grid_folder in grid_folder_list:
        bysubject_outputs = data_comparison[grid_folder].groupby(by="subjects").mean().drop(columns=["fold"])
        bysubject_stats = pd.concat({"mean": bysubject_outputs.mean(), "std": bysubject_outputs.std(ddof=0)}, axis=1)
        byfold_outputs = data_comparison[grid_folder].groupby(by="fold").mean().drop(columns=["subjects"])
        byfold_stats = pd.concat({"mean": byfold_outputs.mean(), "std": byfold_outputs.std(ddof=0)}, axis=1)
        msg = grid_folder
        for key in metric_keys:
            perf_str = " & $%1.1f\pm %1.1f\ (%1.1f)$" % (
                100 * byfold_stats.at[key, "mean"],
                100 * byfold_stats.at[key, "std"],
                100 * bysubject_stats.at[key, "std"])
            msg = msg + perf_str
        msg = msg + "\\\\"
        print(msg)
    # Now compute p-values
    print("\nStatistical tests\n")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        byfold_outputs = data_comparison[grid_folder].groupby(by="fold").mean().drop(columns=["subjects"])
        byfold_outputs_ref = data_comparison[grid_for_reference].groupby(by="fold").mean().drop(columns=["subjects"])
        f1_scores = byfold_outputs.f1.values
        f1_scores_ref = byfold_outputs_ref.f1.values
        pvalue = stats.ttest_ind(f1_scores_ref, f1_scores, equal_var=False)[1]
        print("F1-score (by-fold) %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
            grid_for_reference, f1_scores_ref.mean() * 100, f1_scores_ref.std() * 100,
            grid_folder, f1_scores.mean() * 100, f1_scores.std() * 100,
            pvalue))
    print("")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        bysubject_outputs = data_comparison[grid_folder].groupby(by="subjects").mean().drop(columns=["fold"])
        bysubject_outputs_ref = data_comparison[grid_for_reference].groupby(by="subjects").mean().drop(columns=["fold"])
        f1_scores = bysubject_outputs.f1.values
        f1_scores_ref = bysubject_outputs_ref.f1.values
        pvalue = stats.ttest_ind(f1_scores_ref, f1_scores, equal_var=False)[1]
        print("F1-score (by-subject) %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
            grid_for_reference, f1_scores_ref.mean() * 100, f1_scores_ref.std() * 100,
            grid_folder, f1_scores.mean() * 100, f1_scores.std() * 100,
            pvalue))
    print("")
    # Each subject individually
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        subject_ids = np.unique(data_comparison[grid_folder].subjects.values)
        for subject_id in subject_ids:
            df_curr = data_comparison[grid_folder]
            df_ref = data_comparison[grid_for_reference]
            subject_f1 = df_curr.loc[df_curr['subjects'] == subject_id].f1.values
            subject_f1_ref = df_ref.loc[df_ref['subjects'] == subject_id].f1.values
            pvalue = stats.ttest_ind(subject_f1_ref, subject_f1, equal_var=False)[1]
            print("    S%02d. F1-score %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
                subject_id,
                grid_for_reference, subject_f1_ref.mean() * 100, subject_f1_ref.std() * 100,
                grid_folder, subject_f1.mean() * 100, subject_f1.std() * 100,
                pvalue))
        print("")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        subject_ids = np.unique(data_comparison[grid_folder].subjects.values)
        for subject_id in subject_ids:
            df_curr = data_comparison[grid_folder]
            df_ref = data_comparison[grid_for_reference]
            subject_rec = df_curr.loc[df_curr['subjects'] == subject_id].rec.values
            subject_rec_ref = df_ref.loc[df_ref['subjects'] == subject_id].rec.values
            pvalue = stats.ttest_ind(subject_rec_ref, subject_rec, equal_var=False)[1]
            print("    S%02d. Recall %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
                subject_id,
                grid_for_reference, subject_rec_ref.mean() * 100, subject_rec_ref.std() * 100,
                grid_folder, subject_rec.mean() * 100, subject_rec.std() * 100,
                pvalue))
        print("")
    for grid_folder in grid_folder_list:
        if grid_folder == grid_for_reference:
            continue
        subject_ids = np.unique(data_comparison[grid_folder].subjects.values)
        for subject_id in subject_ids:
            df_curr = data_comparison[grid_folder]
            df_ref = data_comparison[grid_for_reference]
            subject_prec = df_curr.loc[df_curr['subjects'] == subject_id].prec.values
            subject_prec_ref = df_ref.loc[df_ref['subjects'] == subject_id].prec.values
            pvalue = stats.ttest_ind(subject_prec_ref, subject_prec, equal_var=False)[1]
            print("    S%02d. Precision %s(%1.1f+-%1.1f) vs %s(%1.1f+-%1.1f), P=%1.4f" % (
                subject_id,
                grid_for_reference, subject_prec_ref.mean() * 100, subject_prec_ref.std() * 100,
                grid_folder, subject_prec.mean() * 100, subject_prec.std() * 100,
                pvalue))
        print("")

    # Draw figure
    marker_size = 5
    marker_alpha = 0.7
    grid_colors = {
        "v2_time_wave0": viz.GREY_COLORS[6],
        "v2_time_wave1": viz.PALETTE["blue"]
    }
    plot_labels = {
        "v2_time_wave0": "Sin aumento",
        "v2_time_wave1": "Con aumento"
    }
    event_id = "E1SS" if dataset_name == constants.MASS_SS_NAME else "KC"
    ax = axes[2]
    for grid_folder in grid_folder_list:
        byfold_outputs = data_comparison[grid_folder].groupby(by="fold").mean()
        recall = byfold_outputs.rec.values
        precision = byfold_outputs.prec.values
        ax.plot(
            recall, precision, color=grid_colors[grid_folder], linestyle='None', alpha=marker_alpha,
            markeredgewidth=0.0, marker='o', markersize=marker_size, label=plot_labels[grid_folder])
        # ax.set_ylabel('Precision (IoU>%1.1f)' % iou_threshold_report, fontsize=8)
        ax.set_title("Particiones (evento %s)" % event_id, fontsize=8, loc="left")
    ax = axes[3]
    for grid_folder in grid_folder_list:
        bysubject_outputs = data_comparison[grid_folder].groupby(by="subjects").mean()
        recall = bysubject_outputs.rec.values
        precision = bysubject_outputs.prec.values
        ax.plot(
            recall, precision, color=grid_colors[grid_folder], linestyle='None', alpha=marker_alpha,
            markeredgewidth=0.0, marker='o', markersize=marker_size, label=plot_labels[grid_folder])
        ax.set_title("Sujetos (evento %s)" % event_id, fontsize=8, loc="left")

    for ax in axes.flatten():
        plotter.format_precision_recall_plot_simple(
            ax, axis_range=(0.5, 1.0), show_quadrants=False, show_grid=True)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Recall (IoU>%1.1f)" % iou_threshold_report, fontsize=8)
    plt.tight_layout()
    axes[0].text(
        x=-0.01, y=1.15, fontsize=16, s=r"$\bf{A}$",
        ha="left", transform=axes[0].transAxes)
    axes[1].text(
        x=-0.01, y=1.15, fontsize=16, s=r"$\bf{B}$",
        ha="left", transform=axes[1].transAxes)
    axes[2].text(
        x=-0.01, y=1.15, fontsize=16, s=r"$\bf{C}$",
        ha="left", transform=axes[2].transAxes)
    axes[3].text(
        x=-0.01, y=1.15, fontsize=16, s=r"$\bf{D}$",
        ha="left", transform=axes[3].transAxes)
    lines_labels = [axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # plt.subplots_adjust(bottom=0.9)
    leg = fig.legend(
        lines, labels, fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, handletextpad=0.1)
    # leg = axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, frameon=False)

    # Save figure
    fname_prefix = "result_augment_ablation"
    plt.savefig("%s.pdf" % fname_prefix)
    plt.savefig("%s.png" % fname_prefix)
    plt.savefig("%s.svg" % fname_prefix)
    plt.show()
