from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleeprnn.data.dataset import Dataset
from sleeprnn.data import utils, stamp_correction
from sleeprnn.common import constants, pkeys


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


def postprocess_marks(dataset: Dataset, marks, subject_id):
    n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
    pred_marks_n2 = utils.extract_pages_for_stamps(marks, n2_pages, dataset.page_size)
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
    pred_marks_n2 = stamp_correction.combine_close_stamps(
        pred_marks_n2, dataset.fs, min_separation=min_separation)
    pred_marks_n2 = stamp_correction.filter_duration_stamps(
        pred_marks_n2, dataset.fs, min_duration=min_duration, max_duration=max_duration)
    pred_marks_n2 = pred_marks_n2.astype(np.int32)
    return pred_marks_n2
