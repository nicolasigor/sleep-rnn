import os
import pickle

import numpy as np

PATH_THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(PATH_THIS_DIR, '..'))
BASELINES_PATH = os.path.join(PROJECT_ROOT, 'resources', 'comparison_data', 'baselines_2021')

from sleeprnn.helpers import reader
from sleeprnn.common import constants
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT
from sleeprnn.detection import metrics


def get_baseline_predictions(
        baseline_name,
        strategy,
        target_dataset_name,
        target_expert,
        source_dataset_name=None,
        source_expert=None,
):
    baseline_folders = {
        'spinky': '2017_lajnef_spinky',
        'a7': '2019_lacourse_a7',
        'dosed': '2019_chambon_dosed'
    }
    baseline_folder = baseline_folders[baseline_name]

    target_string = '%s_e%d_%s' % (target_dataset_name, target_expert, strategy)
    if source_dataset_name is not None:
        source_string = '%s_e%d_%s' % (source_dataset_name, source_expert, strategy)
    else:
        source_string = target_string
    filename = '%s_from_%s.pkl' % (target_string, source_string)
    filepath = os.path.join(BASELINES_PATH, baseline_folder, filename)
    with open(filepath, 'rb') as handle:
        predictions_dict = pickle.load(handle)
    # output is predictions_dict[fold_id][subject_id]
    return predictions_dict


def get_red_predictions(
        model_version,
        strategy,
        target_dataset,
        target_expert,
        source_dataset_name=None,
        source_expert=None,
        task_mode=constants.N2_RECORD,
        ckpt_folder_prefix='20210529_thesis_indata',
        transfer_date='20210605',
        transfer_desc='sourcestd',
        verbose=False
):
    # Form ckpt_folder
    if source_dataset_name is None:
        ckpt_folder = '%s_%s_e%d_%s_train_%s' % (
            ckpt_folder_prefix, strategy, target_expert, task_mode, target_dataset.dataset_name
        )
    else:
        ckpt_folder = '%s_from_%s_desc_%s_to_%s' % (
            transfer_date,
            '%s_%s_e%d_%s_train_%s' % (
                ckpt_folder_prefix, strategy, source_expert, task_mode, source_dataset_name),
            transfer_desc,
            'e%d_%s_train_%s' % (target_expert, task_mode, target_dataset.dataset_name)
        )
    grid_folder_complete = os.path.join(ckpt_folder, model_version)
    print("Loading predictions from %s" % grid_folder_complete) if verbose else None
    predictions_dict = reader.read_predictions_crossval(grid_folder_complete, target_dataset, task_mode)
    # Ensure optimal threshold in predictions
    opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[grid_folder_complete]
    for k in predictions_dict.keys():
        for set_name in predictions_dict[k].keys():
            predictions_dict[k][set_name].set_probability_threshold(opt_thr_list[k])
    # output is predictions_dict[fold_id][subset] -> [subject_id]
    return predictions_dict


def compute_fold_performance(events_list, detections_list, average_mode, iou_threshold_report=0.2):
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}
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
    outputs = {
        'F1-score': f1_score[0],
        'Recall': recall[0],
        'Precision': precision[0],
        'mIoU': miou
    }
    return outputs


def compute_subject_performance(events_list, detections_list, iou_threshold_report=0.2):
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
    miou_list = np.array(miou_list)
    outputs = {
        'F1-score': f1_score[:, 0],
        'Recall': recall[:, 0],
        'Precision': precision[:, 0],
        'mIoU': miou_list
    }
    return outputs


def format_metric(mean_value, std_value, scale_by=100):
    return "$%1.1f\pm %1.1f$" % (scale_by * mean_value, scale_by * std_value)
