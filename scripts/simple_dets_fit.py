from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import json
import os
import pickle
from pprint import pprint
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.detection import simple_detection
from sleeprnn.detection import metrics
from figs_thesis.fig_utils import compute_fold_performance

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == "__main__":
    dataset_name = constants.MODA_SS_NAME
    average_mode = constants.MICRO_AVERAGE

    detector_to_fit = 'relative'
    search_space = np.arange(3, 5 + 0.001, 0.1)

    dataset = load_dataset(dataset_name)

    average_metric_fn_dict = {
        constants.MACRO_AVERAGE: metrics.average_metric_macro_average,
        constants.MICRO_AVERAGE: metrics.average_metric_micro_average}
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}

    print("fitting simple detector '%s'" % detector_to_fit)
    af1_list = []
    for thr in search_space:
        print("Testing thr %s" % thr)
        events_list = []
        detections_list = []
        for subject_id in dataset.all_ids:
            signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
            n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
            if detector_to_fit == 'absolute':
                detections = simple_detection.simple_detector_absolute(
                    signal, dataset.fs, thr)
            elif detector_to_fit == 'relative':
                detections = simple_detection.simple_detector_relative(
                    signal, dataset.fs, thr, n2_pages, dataset.page_duration)
            else:
                raise ValueError
            # detections only in N2
            detections = utils.extract_pages_for_stamps(detections, n2_pages, dataset.page_size)
            # events only in N2
            events = dataset.get_subject_stamps(subject_id, pages_subset=constants.N2_RECORD)
            events_list.append(events)
            detections_list.append(detections)
        # measure performance
        af1 = average_metric_fn_dict[average_mode](events_list, detections_list)
        af1_list.append(af1)
    # find best thr
    max_idx = np.argmax(af1_list).item()
    best_thr = search_space[max_idx]
    print("Best thr found: %s" % best_thr)

    # performance of best thr in entire dataset (overfitting)
    events_list = []
    detections_list = []
    for subject_id in dataset.all_ids:
        signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
        n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
        if detector_to_fit == 'absolute':
            detections = simple_detection.simple_detector_absolute(
                signal, dataset.fs, best_thr)
        elif detector_to_fit == 'relative':
            detections = simple_detection.simple_detector_relative(
                signal, dataset.fs, best_thr, n2_pages, dataset.page_duration)
        else:
            raise ValueError
        # detections only in N2
        detections = utils.extract_pages_for_stamps(detections, n2_pages, dataset.page_size)
        # events only in N2
        events = dataset.get_subject_stamps(subject_id, pages_subset=constants.N2_RECORD)
        events_list.append(events)
        detections_list.append(detections)
    # measure performance of best thr
    results = compute_fold_performance(events_list, detections_list, average_mode)
    for metric_name in results.keys():
        print('%s: %1.2f' % (
            metric_name.ljust(20),
            100 * results[metric_name]
        ))


