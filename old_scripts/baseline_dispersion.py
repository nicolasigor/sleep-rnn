from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.detection import metrics
from sleeprnn.common import constants, viz, pkeys
from sleeprnn.data.mass_ss import IDS_TEST
from sleeprnn.helpers import reader
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT

RESULTS_PATH = os.path.join(project_root, 'results')
BASELINES_PATH = os.path.join(project_root, 'resources', 'comparison_data', 'baselines')


if __name__ == '__main__':
    dataset_name = constants.MASS_KC_NAME
    which_expert = 1

    baseline_name_dict = {
        constants.MASS_SS_NAME: ['2019_chambon', '2019_lacourse'],
        constants.MASS_KC_NAME: ['2019_chambon', '2017_lajnef']
    }
    baseline_print_name_dict = {
        constants.MASS_SS_NAME: ['DOSED', 'A7'],
        constants.MASS_KC_NAME: ['DOSED', 'SPINKY']
    }

    baseline_name_list = baseline_name_dict[dataset_name]
    baseline_print_name_list = baseline_print_name_dict[dataset_name]
    iou_to_show = 0.2
    task_mode = constants.N2_RECORD
    test_ids = IDS_TEST.copy()
    n_folds = 10
    axis_lims = [0, 1]

    for baseline_name, baseline_print_name in zip(baseline_name_list, baseline_print_name_list):
        folder_to_check = os.path.join(
            BASELINES_PATH, baseline_name, dataset_name, 'e%d' % which_expert)
        if os.path.exists(folder_to_check):
            print('%s found. ' % baseline_name, end='', flush=True)
            right_prefix = '%s_%s_e%d' % (baseline_name, dataset_name, which_expert)
            pr_data = {}
            for subject_id in test_ids:
                rec_list = []
                pre_list = []
                for k in range(n_folds):
                    fname = '%s_fold%d_s%02d.npz' % (
                        right_prefix, k, subject_id)
                    fname_path = os.path.join(
                        folder_to_check, 'fold%d' % k, fname)
                    this_data = np.load(fname_path)
                    iou_axis = this_data['iou_axis']
                    rec = this_data['recall_vs_iou']
                    pre = this_data['precision_vs_iou']
                    idx_iou = np.argmin((iou_axis - iou_to_show) ** 2).item()
                    rec_list.append(rec[idx_iou])
                    pre_list.append(pre[idx_iou])
                pr_data[subject_id] = {
                    'recall': np.array(rec_list),
                    'precision': np.array(pre_list)
                }
            # Plot precision recall curve
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
            colors = [viz.PALETTE['red'], viz.PALETTE['blue'], viz.PALETTE['green'], viz.PALETTE['dark']]
            for i,subject_id in enumerate(test_ids):
                ax.plot(
                    pr_data[subject_id]['recall'],
                    pr_data[subject_id]['precision'],
                    markersize=3,
                    linestyle="none",
                    marker='o',
                    color=colors[i],
                    label='S%02d' % subject_id)
            ax.legend(loc='lower left', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('Recall (IoU > %1.1f)' % iou_to_show, fontsize=8)
            ax.set_ylabel('Precision (IoU > %1.1f)' % iou_to_show, fontsize=8)
            ax.set_title('%s (%s-%s-E%s)\nTest Subjects, %d training folds' % (
                baseline_print_name,
                "-".join(dataset_name.upper().split("_")),
                task_mode.upper(),
                which_expert, n_folds), fontsize=8)
            ax.set_xlim(axis_lims)
            ax.set_ylim(axis_lims)
            ax.plot(
                axis_lims, axis_lims,
                linewidth=1, color=viz.GREY_COLORS[4],
                zorder=1)
            plt.tight_layout()
            plt.savefig(
                '%s_e%d_%s.png' % (dataset_name, which_expert, baseline_name),
                bbox_inches="tight", dpi=200)
            plt.show()

    # RED
    models = {
        'v11': {
            'print_name': 'RED-Time',
            'ckpt': {
                'ss_e1': '20191227_bsf_10runs_e1_n2_train_mass_ss/v11',
                'ss_e2': '20191227_bsf_10runs_e2_n2_train_mass_ss/v11',
                'kc_e1': '20191227_bsf_10runs_e1_n2_train_mass_kc/v11'
            }
        },
        'v19': {
            'print_name': 'RED-CWT',
            'ckpt': {
                'ss_e1': '20191227_bsf_10runs_e1_n2_train_mass_ss/v19',
                'ss_e2': '20191227_bsf_10runs_e2_n2_train_mass_ss/v19',
                'kc_e1': '20191227_bsf_10runs_e1_n2_train_mass_kc/v19'
            }
        }

    }
    seed_id_list = [i for i in range(n_folds)]
    dataset = reader.load_dataset(dataset_name, params={pkeys.FS: 200})
    for model in models.keys():
        print_name = models[model]['print_name']
        ckpt_display_dict = models[model]['ckpt']
        key = "%s_e%d" % (dataset_name.split("_")[1], which_expert)
        ckpt_folder = ckpt_display_dict[key]
        predictions_dict = reader.read_prediction_with_seeds(
            ckpt_folder, dataset_name, task_mode, seed_id_list,
            set_list=[constants.TEST_SUBSET], parent_dataset=dataset)
        optimal_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[ckpt_folder]
        pr_data = {}
        for subject_id in test_ids:
            rec_list = []
            pre_list = []
            events = dataset.get_subject_stamps(
                subject_id,
                which_expert=which_expert,
                pages_subset=task_mode)
            for k in range(n_folds):
                t_preds = predictions_dict[k][constants.TEST_SUBSET]
                t_preds.set_probability_threshold(optimal_thr_list[k])
                detections = t_preds.get_subject_stamps(subject_id)
                # Matching
                iou_matchings, idx_matchings = metrics.matching(
                    events, detections)
                # Measure stuff
                seed_recall_vs_iou = metrics.metric_vs_iou(
                    events, detections, [iou_to_show],
                    iou_matching=iou_matchings,
                    metric_name=constants.RECALL)
                seed_precision_vs_iou = metrics.metric_vs_iou(
                    events, detections, [iou_to_show],
                    iou_matching=iou_matchings,
                    metric_name=constants.PRECISION)
                rec_list.append(seed_recall_vs_iou[0])
                pre_list.append(seed_precision_vs_iou[0])
            pr_data[subject_id] = {
                'recall': np.array(rec_list),
                'precision': np.array(pre_list)
            }
        # Plot precision recall curve
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        colors = [viz.PALETTE['red'], viz.PALETTE['blue'],
                  viz.PALETTE['green'], viz.PALETTE['dark']]
        for i, subject_id in enumerate(test_ids):
            ax.plot(
                pr_data[subject_id]['recall'],
                pr_data[subject_id]['precision'],
                markersize=3,
                linestyle="none",
                marker='o',
                color=colors[i],
                label='S%02d' % subject_id)
        ax.legend(loc='lower left', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('Recall (IoU > %1.1f)' % iou_to_show, fontsize=8)
        ax.set_ylabel('Precision (IoU > %1.1f)' % iou_to_show, fontsize=8)
        ax.set_title('%s (%s-%s-E%s)\nTest Subjects, %d training folds' % (
            print_name,
            "-".join(dataset_name.upper().split("_")),
            task_mode.upper(),
            which_expert, n_folds), fontsize=8)
        ax.set_xlim(axis_lims)
        ax.set_ylim(axis_lims)
        ax.plot(
            axis_lims, axis_lims,
            linewidth=1, color=viz.GREY_COLORS[4],
            zorder=1)
        plt.tight_layout()
        plt.savefig(
            '%s_e%d_%s.png' % (dataset_name, which_expert, model),
            bbox_inches="tight", dpi=200)
        plt.show()





