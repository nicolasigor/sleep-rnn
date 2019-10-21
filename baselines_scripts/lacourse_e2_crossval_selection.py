from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.data import utils
from sleeprnn.detection import metrics
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(project_root, '../sleep-baselines/2019_lacourse'))

if __name__ == '__main__':

    dataset_name = constants.MASS_SS_NAME
    which_expert = 2
    dataset_params = {pkeys.FS: 128}

    task_mode = constants.N2_RECORD
    id_try_list = [0, 1, 2, 3]

    # Load expert annotations 
    dataset = load_dataset(
        dataset_name,
        params=dataset_params, load_checkpoint=False)
    all_train_ids = dataset.train_ids
    page_size = dataset.page_size
    print('Page size:', page_size)
    print('All train ids', all_train_ids)
    gs_dict = {}
    n2_dict = {}
    for subject_id in all_train_ids:
        n2_dict[subject_id] = dataset.get_subject_pages(
            subject_id,
            pages_subset=task_mode)

    # Load predictions
    pred_folder = os.path.join(BASELINE_PATH, 'output_%s' % dataset_name, 'e%d' % which_expert)
    print('Loading predictions from %s' % pred_folder, flush=True)
    pred_files = os.listdir(pred_folder)
    pred_dict = {}
    visited_settings = []
    for file in pred_files:
        subject_id = int(file.split('_')[3][1:])
        setting = '_'.join(file.split('_')[4:])[:-4]
        if setting not in visited_settings:
            pred_dict[setting] = {}
            visited_settings.append(setting)
        # sample marks
        filepath = os.path.join(pred_folder, file)
        pred_data = pd.read_csv(filepath, sep='\t')
        # We substract 1 to translate from matlab to numpy indexing system
        start_samples = pred_data.start_sample.values - 1
        end_samples = pred_data.end_sample.values - 1
        pred_marks = np.stack([start_samples, end_samples], axis=1)
        # Valid subset of marks
        pred_marks_n2 = utils.extract_pages_for_stamps(
            pred_marks, n2_dict[subject_id], page_size)
        # Save marks for evaluation
        pred_dict[setting][subject_id] = pred_marks_n2

    # Start evaluation
    print('Starting evaluation of %d settings' % len(visited_settings), flush=True)
    for k in id_try_list:
        print('Using fold %d' % k, flush=True)
        train_ids, _ = utils.split_ids_list_v2(all_train_ids, split_id=k)
        train_ids.sort()
        train_marks_list = dataset.get_subset_stamps(
            train_ids,
            which_expert=which_expert,
            pages_subset=task_mode)
        setting_performance = []
        for setting in visited_settings:
            # Create subset of marks
            pred_marks_list = []
            for subject_id in train_ids:
                pred_marks_list.append(pred_dict[setting][subject_id])
            af1_of_setting = metrics.average_metric_with_list(
                train_marks_list, pred_marks_list, verbose=False)
            setting_performance.append(af1_of_setting)
        # Look for best performance
        max_idx = np.argmax(setting_performance).item()
        max_af1 = setting_performance[max_idx]
        max_setting = visited_settings[max_idx]
        print('Best AF1 %1.4f with setting %s' % (max_af1, max_setting), flush=True)
