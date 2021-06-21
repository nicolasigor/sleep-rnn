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
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    search_space_high = [2.9]  # np.arange(8, 12 + 0.001, 1)
    search_space_low_factor = np.arange(0.7, 0.82 + 0.001, 0.02)

    dataset = load_dataset(dataset_name)

    average_metric_fn_dict = {
        constants.MACRO_AVERAGE: metrics.average_metric_macro_average,
        constants.MICRO_AVERAGE: metrics.average_metric_micro_average}
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}

    print("fitting simple detector '%s'" % detector_to_fit)
    search_space = list(itertools.product(search_space_high, search_space_low_factor))
    n_combs = len(search_space)
    print("Search space high:", search_space_high)
    print("Search space low:", search_space_low_factor)
    print("Combinations to try: %d" % n_combs)

    af1_list = []
    for i_comb, combination in enumerate(search_space):
        thr_high, thr_low_factor = combination
        print("Testing thr high %1.2f and low factor %1.2f (%d / %d)." % (thr_high, thr_low_factor, i_comb + 1, n_combs), end='')
        events_list = []
        detections_list = []
        for subject_id in dataset.all_ids:
            signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
            n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
            if detector_to_fit == 'absolute':
                detections = simple_detection.simple_detector_absolute(
                    signal, dataset.fs, thr_high,
                    amplitude_low_thr_factor=thr_low_factor)
            elif detector_to_fit == 'relative':
                detections = simple_detection.simple_detector_relative(
                    signal, dataset.fs, thr_high, n2_pages, dataset.page_duration,
                    amplitude_low_thr_factor=thr_low_factor)
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
        print(" AF1 %1.2f" % (100 * af1))
    # find best thr
    max_idx = np.argmax(af1_list).item()
    best_thr_high, best_thr_low_factor = search_space[max_idx]
    print("Best combination found found: high %1.2f, low factor %1.2f (AF1 %1.2f)" % (
        best_thr_high, best_thr_low_factor, 100 * np.max(af1_list)))

    # performance of best thr in entire dataset (overfitting)
    events_list = []
    detections_list = []
    for subject_id in dataset.all_ids:
        signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
        n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
        if detector_to_fit == 'absolute':
            detections = simple_detection.simple_detector_absolute(
                signal, dataset.fs, best_thr_high,
                amplitude_low_thr_factor=best_thr_low_factor)
        elif detector_to_fit == 'relative':
            detections = simple_detection.simple_detector_relative(
                signal, dataset.fs, best_thr_high, n2_pages, dataset.page_duration,
                amplitude_low_thr_factor=best_thr_low_factor)
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
        print('%s: %1.1f' % (
            metric_name.ljust(20),
            100 * results[metric_name]
        ))


