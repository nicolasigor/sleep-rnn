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
    dataset_name = constants.CAP_SS_NAME

    dataset = load_dataset(dataset_name)

    thr_abs_high = 10
    thr_abs_low = 0.86
    thr_rel_high = 2.9
    thr_rel_low = 0.80

    save_dir_parent = os.path.join(project_root, 'resources', 'datasets', 'simple_detections', dataset_name)
    save_dir_abs = os.path.join(save_dir_parent, 'spindle_s1_abs')
    save_dir_rel = os.path.join(save_dir_parent, 'spindle_s2_rel')
    os.makedirs(save_dir_abs, exist_ok=True)
    os.makedirs(save_dir_rel, exist_ok=True)

    subject_ids = dataset.all_ids
    for i_sub, subject_id in enumerate(subject_ids):
        print("Predicting subject %s (%d / %d)" % (subject_id, i_sub + 1, len(subject_ids)))
        signal = dataset.get_subject_signal(subject_id, normalize_clip=False, which_expert=1)
        n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)

        detections_abs = simple_detection.simple_detector_absolute(
            signal, dataset.fs, thr_abs_high, amplitude_low_thr_factor=thr_abs_low)
        detections_abs = utils.extract_pages_for_stamps(detections_abs, n2_pages, dataset.page_size)

        detections_rel = simple_detection.simple_detector_relative(
            signal, dataset.fs, thr_rel_high, n2_pages, dataset.page_duration, amplitude_low_thr_factor=thr_rel_low)
        detections_rel = utils.extract_pages_for_stamps(detections_rel, n2_pages, dataset.page_size)

        fname_abs = os.path.join(
            save_dir_abs, 'SimpleDetectionAbsolute_s%s_thr%d-%1.2f_fs%d.txt' % (
                subject_id, thr_abs_high, thr_abs_low, dataset.fs))
        fname_rel = os.path.join(
            save_dir_rel, 'SimpleDetectionRelative_s%s_thr%1.1f-%1.2f_fs%d.txt' % (
                subject_id, thr_rel_high, thr_rel_low, dataset.fs))
        np.savetxt(fname_abs, detections_abs, delimiter=',', fmt="%d")
        np.savetxt(fname_rel, detections_rel, delimiter=',', fmt="%d")
