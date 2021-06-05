from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from joblib import delayed, Parallel
import os
import pickle

import numpy as np

from sleeprnn.data.dataset import Dataset
from sleeprnn.data import utils, stamp_correction
from sleeprnn.common import constants, pkeys
from sleeprnn.detection import metrics, postprocessing


def get_partitions(dataset: Dataset, strategy, n_seeds):
    train_ids_list = []
    val_ids_list = []
    test_ids_list = []
    if strategy == 'fixed':
        for fold_id in range(n_seeds):
            train_ids, val_ids = utils.split_ids_list_v2(dataset.train_ids, split_id=fold_id)
            train_ids_list.append(train_ids)
            val_ids_list.append(val_ids)
            test_ids_list.append(dataset.test_ids)
    elif strategy == '5cv':
        for cv_seed in range(n_seeds):
            for fold_id in range(5):
                train_ids, val_ids, test_ids = dataset.cv_split(5, fold_id, cv_seed)
                train_ids_list.append(train_ids)
                val_ids_list.append(val_ids)
                test_ids_list.append(test_ids)
    else:
        raise ValueError
    return train_ids_list, val_ids_list, test_ids_list


def get_settings(predictions_dir, extract_setting_fn):
    subject_folders = os.listdir(predictions_dir)
    subject_folder = subject_folders[0]
    subject_path = os.path.join(predictions_dir, subject_folder)
    detection_files = os.listdir(subject_path)
    settings = [extract_setting_fn(d) for d in detection_files]
    settings = np.asarray(settings)
    return settings


def postprocess_marks(dataset: Dataset, marks, subject_id, apply_temporal_processing=True, kc_split=False):
    if apply_temporal_processing:
        if dataset.event_name == constants.SPINDLE:
            if 'inta' in dataset.dataset_name:
                min_separation = 0.5
                min_duration = 0.5
                max_duration = 5.0
            else:
                min_separation = 0.3
                min_duration = 0.3
                max_duration = 3.0
        else:
            min_separation = None
            min_duration = 0.3
            max_duration = None
        marks = stamp_correction.combine_close_stamps(
            marks, dataset.fs, min_separation=min_separation)
        marks = stamp_correction.filter_duration_stamps(
            marks, dataset.fs, min_duration=min_duration, max_duration=max_duration)
    if kc_split:
        signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
        marks = postprocessing.kcomplex_stamp_split(signal, marks, dataset.fs)
    n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
    pred_marks_n2 = utils.extract_pages_for_stamps(marks, n2_pages, dataset.page_size)
    pred_marks_n2 = pred_marks_n2.astype(np.int32)
    return pred_marks_n2


def get_prediction_dict(dataset: Dataset, predictions_dir, settings, get_raw_marks_fn):
    pred_dict = {}
    for setting in settings:
        pred_dict[setting] = {}
        for subject_id in dataset.all_ids:
            pred_marks = get_raw_marks_fn(predictions_dir, subject_id, setting, dataset)
            pred_marks_n2 = postprocess_marks(dataset, pred_marks, subject_id)
            pred_dict[setting][subject_id] = pred_marks_n2
    return pred_dict


def get_metric_functions(average_mode):
    average_metric_fn_dict = {
        constants.MACRO_AVERAGE: metrics.average_metric_macro_average,
        constants.MICRO_AVERAGE: metrics.average_metric_micro_average}
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}
    return metric_vs_iou_fn_dict[average_mode], average_metric_fn_dict[average_mode]


def train_grid(dataset: Dataset, which_expert, train_ids_list, predictions_dict, average_mode):
    fitted_setting_dict = {}
    n_folds = len(train_ids_list)
    settings = np.array(list(predictions_dict.keys()))
    _, average_metric_fn = get_metric_functions(average_mode)
    for fold_id in range(n_folds):
        print('Training fold %d. ' % fold_id, flush=True, end='')
        train_ids = train_ids_list[fold_id]
        events_list = dataset.get_subset_stamps(
            train_ids, which_expert=which_expert, pages_subset=constants.N2_RECORD)
        predictions_at_setting_list = []
        for setting in settings:
            detections_list = [predictions_dict[setting][subject_id] for subject_id in train_ids]
            predictions_at_setting_list.append(detections_list)
        af1_list = Parallel(n_jobs=-1)(
            delayed(average_metric_fn)(events_list, detections_list)
            for detections_list in predictions_at_setting_list)
        af1_list = np.array(af1_list)
        max_idx = np.argmax(af1_list).item()
        max_af1_value = af1_list[max_idx]
        max_setting = settings[max_idx]
        print('Best Train AF1 %1.2f with setting %s' % (max_af1_value, max_setting), flush=True)
        fitted_setting_dict[fold_id] = max_setting.item()
    return fitted_setting_dict


def evaluate_by_fold(
        dataset: Dataset, which_expert, test_ids_list,
        fitted_setting_dict, predictions_dict, average_mode, iou_threshold_report):
    # By fold metrics
    outputs = {'af1': [], 'f1': [], 'rec': [], 'prec': [], 'miou': []}
    metric_vs_iou_fn, average_metric_fn = get_metric_functions(average_mode)
    n_folds = len(test_ids_list)
    for fold_id in range(n_folds):
        best_setting = fitted_setting_dict[fold_id]
        test_ids = test_ids_list[fold_id]
        events_list = dataset.get_subset_stamps(
            test_ids, which_expert=which_expert, pages_subset=constants.N2_RECORD)
        detections_list = [predictions_dict[best_setting][subject_id] for subject_id in test_ids]
        iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
        af1_best = average_metric_fn(
            events_list, detections_list, iou_matching_list=iou_matching_list)
        f1_best = metric_vs_iou_fn(
            events_list, detections_list, [iou_threshold_report], iou_matching_list=iou_matching_list)
        precision_best = metric_vs_iou_fn(
            events_list, detections_list, [iou_threshold_report],
            metric_name=constants.PRECISION, iou_matching_list=iou_matching_list)
        recall_best = metric_vs_iou_fn(
            events_list, detections_list, [iou_threshold_report],
            metric_name=constants.RECALL, iou_matching_list=iou_matching_list)
        nonzero_iou_list = [iou_matching[iou_matching > 0] for iou_matching in iou_matching_list]
        if average_mode == constants.MACRO_AVERAGE:
            miou_list = [np.mean(nonzero_iou) for nonzero_iou in nonzero_iou_list]
            miou = np.mean(miou_list)
        elif average_mode == constants.MICRO_AVERAGE:
            miou = np.concatenate(nonzero_iou_list).mean()
        else:
            raise ValueError("Average mode %s invalid" % average_mode)
        outputs['af1'].append(af1_best)
        outputs['f1'].append(f1_best)
        outputs['prec'].append(precision_best)
        outputs['rec'].append(recall_best)
        outputs['miou'].append(miou)
    for key in outputs.keys():
        outputs[key] = np.array(outputs[key])
    return outputs


def evaluate_dosed_by_fold(
        dataset: Dataset, which_expert, test_ids_list,
        predictions_dict, average_mode, iou_threshold_report):
    # By fold metrics
    outputs = {
        'AF1      ': [],
        'F1       ': [],
        'Recall   ': [],
        'Precision': [],
        'mIoU     ': []}
    metric_vs_iou_fn, average_metric_fn = get_metric_functions(average_mode)
    n_folds = len(test_ids_list)
    for fold_id in range(n_folds):
        test_ids = test_ids_list[fold_id]
        events_list = dataset.get_subset_stamps(
            test_ids, which_expert=which_expert, pages_subset=constants.N2_RECORD)
        detections_list = [predictions_dict[fold_id][subject_id] for subject_id in test_ids]
        iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
        af1_best = average_metric_fn(
            events_list, detections_list, iou_matching_list=iou_matching_list)
        f1_best = metric_vs_iou_fn(
            events_list, detections_list, [iou_threshold_report], iou_matching_list=iou_matching_list)
        precision_best = metric_vs_iou_fn(
            events_list, detections_list, [iou_threshold_report],
            metric_name=constants.PRECISION, iou_matching_list=iou_matching_list)
        recall_best = metric_vs_iou_fn(
            events_list, detections_list, [iou_threshold_report],
            metric_name=constants.RECALL, iou_matching_list=iou_matching_list)
        nonzero_iou_list = [iou_matching[iou_matching > 0] for iou_matching in iou_matching_list]
        if average_mode == constants.MACRO_AVERAGE:
            miou_list = [np.mean(nonzero_iou) for nonzero_iou in nonzero_iou_list]
            miou = np.mean(miou_list)
        elif average_mode == constants.MICRO_AVERAGE:
            miou = np.concatenate(nonzero_iou_list).mean()
        else:
            raise ValueError("Average mode %s invalid" % average_mode)
        outputs['AF1      '].append(af1_best)
        outputs['F1       '].append(f1_best)
        outputs['Recall   '].append(recall_best)
        outputs['Precision'].append(precision_best)
        outputs['mIoU     '].append(miou)
    for key in outputs.keys():
        outputs[key] = np.array(outputs[key])
    return outputs


def print_performance(results):
    report_dict = {}
    for key in results:
        report_dict[key] = {'mean': 100 * np.mean(results[key]), 'std': 100 * np.std(results[key])}
    # Regular printing
    for key in results:
        print("%s: %1.2f \pm %1.2f" % (key, report_dict[key]['mean'], report_dict[key]['std']))


def get_raw_dosed_marks_in_fold(predictions_dir, fold_id, dataset: Dataset, print_thr=True):
    pred_filepath = os.path.join(predictions_dir, 'fold%02d_dosed_predictions_ckpt.pkl' % fold_id)
    with open(pred_filepath, 'rb') as handle:
        this_predictions = pickle.load(handle)
    if print_thr:
        thr_filepath = os.path.join(predictions_dir, 'fold%02d_dosed_threshold_ckpt.pkl' % fold_id)
        with open(thr_filepath, 'rb') as handle:
            this_thr = pickle.load(handle)
        print("Fold %02d with threshold %1.4f" % (fold_id, this_thr))
    records = list(this_predictions.keys())
    fs_predicted = int(records[0].split(".")[0].split("_")[3])
    fs_target = dataset.fs
    pred_dict = {}
    for record_name in records:
        subject_id = record_name.split(".")[0].split("_")[1][1:]
        dataset_name_short = record_name.split(".")[0].split("_")[0]
        if dataset_name_short in ['mass', 'inta']:
            subject_id = int(subject_id)
        pred_marks = np.array(this_predictions[record_name][0])
        pred_marks = (pred_marks * fs_target / fs_predicted).astype(np.int32)
        pred_dict[subject_id] = pred_marks
    return pred_dict


def get_raw_dosed_marks(predictions_dir, n_folds, dataset: Dataset, print_thr=True):
    pred_dict = {}
    for fold_id in range(n_folds):
        pred_dict[fold_id] = get_raw_dosed_marks_in_fold(predictions_dir, fold_id, dataset, print_thr=print_thr)
    return pred_dict


def postprocess_dosed_marks(dataset: Dataset, prediction_dict):
    kc_split = (dataset.event_name == constants.KCOMPLEX)
    new_prediction_dict = {}
    for fold_id in prediction_dict.keys():
        new_prediction_dict[fold_id] = {}
        for subject_id in prediction_dict[fold_id].keys():
            new_prediction_dict[fold_id][subject_id] = postprocess_marks(
                dataset, prediction_dict[fold_id][subject_id], subject_id, kc_split=kc_split)
    return new_prediction_dict
