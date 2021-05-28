from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from joblib import delayed, Parallel
import os
from pprint import pprint
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.data import utils, stamp_correction
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines/2019_lacourse_a7'))


def get_settings(parent_dir):
    subject_folders = os.listdir(parent_dir)
    subject_folder = subject_folders[0]
    subject_path = os.path.join(parent_dir, subject_folder)
    detection_files = os.listdir(subject_path)
    settings = ["_".join(d.split("_")[2:])[:-4] for d in detection_files]
    settings = np.asarray(settings)
    return settings


def get_marks(
        subject_id,
        setting,
        min_separation=None,
        min_duration=None,
        max_duration=None,
        pages_subset=constants.N2_RECORD
):
    filepath = os.path.join(
        pred_folder,
        's%s' % subject_id,
        'EventDetection_s%s_%s.txt' % (subject_id, setting))
    pred_data = pd.read_csv(filepath, sep='\t')
    # We substract 1 to translate from matlab to numpy indexing system
    start_samples = pred_data.start_sample.values - 1
    end_samples = pred_data.end_sample.values - 1
    pred_marks = np.stack([start_samples, end_samples], axis=1)
    # Valid subset of marks
    n2_pages = dataset.get_subject_pages(subject_id, pages_subset=pages_subset)
    pred_marks_n2 = utils.extract_pages_for_stamps(pred_marks, n2_pages, dataset.page_size)
    # Postprocessing
    pred_marks_n2 = stamp_correction.combine_close_stamps(
        pred_marks_n2, dataset.fs, min_separation=min_separation)
    pred_marks_n2 = stamp_correction.filter_duration_stamps(
        pred_marks_n2, dataset.fs, min_duration=min_duration, max_duration=max_duration)
    pred_marks_n2 = pred_marks_n2.astype(np.int32)
    return pred_marks_n2


def get_prediction_dict(
        settings,
        min_separation=None,
        min_duration=None,
        max_duration=None,
):
    pred_dict = {}
    for setting in settings:
        pred_dict[setting] = {}
        for subject_id in dataset.all_ids:
            predicted = get_marks(
                subject_id, setting,
                min_separation=min_separation, min_duration=min_duration, max_duration=max_duration)
            pred_dict[setting][subject_id] = predicted
    return pred_dict


def get_partitions(strategy, n_seeds):
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


if __name__ == '__main__':
    # Dataset evaluation settings
    dataset_name = constants.INTA_SS_NAME
    which_expert = 1
    strategy = '5cv'
    n_seeds = 3
    min_separation = 0.5
    min_duration = 0.5
    max_duration = 5.0
    average_mode = constants.MACRO_AVERAGE
    iou_threshold_report = 0.2

    # ------------------------------------------------------

    # Load dataset
    dataset = reader.load_dataset(dataset_name, verbose=False)

    # Predictions
    pred_folder = os.path.join(BASELINE_PATH, 'output_thesis_%s' % dataset_name.split("_")[0])
    print('Loading predictions from %s' % pred_folder, flush=True)
    settings = get_settings(pred_folder)
    pred_dict = get_prediction_dict(
        settings, min_separation=min_separation, min_duration=min_duration, max_duration=max_duration)

    # Evaluation
    average_metric_fn_dict = {
        constants.MACRO_AVERAGE: metrics.average_metric_macro_average,
        constants.MICRO_AVERAGE: metrics.average_metric_micro_average}
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}
    train_ids_list, _, test_ids_list = get_partitions(strategy, n_seeds)
    n_folds = len(train_ids_list)
    # Fitting
    fitted_setting_dict = {}
    for fold_id in range(n_folds):
        print('Using fold %d. ' % fold_id, flush=True, end='')
        train_ids = train_ids_list[fold_id]
        events_list = dataset.get_subset_stamps(
            train_ids, which_expert=which_expert, pages_subset=constants.N2_RECORD)
        predictions_at_setting_list = []
        for setting in settings:
            detections_list = [pred_dict[setting][subject_id] for subject_id in train_ids]
            predictions_at_setting_list.append(detections_list)
        af1_list = Parallel(n_jobs=-1)(
            delayed(average_metric_fn_dict[average_mode])(events_list, detections_list)
            for detections_list in predictions_at_setting_list)
        af1_list = np.array(af1_list)
        max_af1_value = np.max(af1_list)
        max_locs = np.where(af1_list == max_af1_value)[0]
        possible_settings = settings[max_locs]
        chosen_setting = possible_settings[0]
        # max_idx = np.argmax(af1_list).item()
        # max_af1 = af1_list[max_idx]
        # max_setting = settings[max_idx]
        print('Best Train AF1 %1.2f with setting %s' % (max_af1_value, possible_settings), flush=True)
        fitted_setting_dict[fold_id] = chosen_setting
    # Test performance (by-fold mean and dispersion)
    outputs = {'af1_best': [], 'f1_best': [], 'prec_best': [], 'rec_best': []}
    for fold_id in range(n_folds):
        best_setting = fitted_setting_dict[fold_id]
        test_ids = test_ids_list[fold_id]
        events_list = dataset.get_subset_stamps(
            test_ids, which_expert=which_expert, pages_subset=constants.N2_RECORD)
        detections_list = [pred_dict[best_setting][subject_id] for subject_id in test_ids]
        iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
        af1_best = average_metric_fn_dict[average_mode](
            events_list, detections_list, iou_matching_list=iou_matching_list)
        f1_best = metric_vs_iou_fn_dict[average_mode](
            events_list, detections_list, [iou_threshold_report], iou_matching_list=iou_matching_list)
        precision_best = metric_vs_iou_fn_dict[average_mode](
            events_list, detections_list, [iou_threshold_report],
            metric_name=constants.PRECISION, iou_matching_list=iou_matching_list)
        recall_best = metric_vs_iou_fn_dict[average_mode](
            events_list, detections_list, [iou_threshold_report],
            metric_name=constants.RECALL, iou_matching_list=iou_matching_list)
        outputs['af1_best'].append(af1_best)
        outputs['f1_best'].append(f1_best)
        outputs['prec_best'].append(precision_best)
        outputs['rec_best'].append(recall_best)
    report_dict = {}
    for key in outputs:
        report_dict[key] = {'mean': 100 * np.mean(outputs[key]), 'std': 100 * np.std(outputs[key])}
    str_to_show = (
        'AF1 %1.2f/%1.2f, '
        'F %1.2f/%1.2f, '
        'P %1.1f/%s, '
        'R %1.1f/%s'
        % (report_dict['af1_best']['mean'], report_dict['af1_best']['std'],
           report_dict['f1_best']['mean'], report_dict['f1_best']['std'],
           report_dict['prec_best']['mean'],
           ('%1.1f' % report_dict['prec_best']['std']).rjust(4),
           report_dict['rec_best']['mean'],
           ('%1.1f' % report_dict['rec_best']['std']).rjust(4),
           ))
    eval_id = "%s_exp%d" % (dataset_name, which_expert)
    print("Test performance (%s, iou >= %1.2f, from train to test) for %s" % (
        average_mode, iou_threshold_report, eval_id))
    print(str_to_show)
