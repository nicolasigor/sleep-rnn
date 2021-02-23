from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import itertools
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
from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


def split_train(x_train, y_train, page_size, border_size):
    n_train = x_train.shape[0]
    # Remove to recover single page from augmented page
    remove_size = border_size + page_size // 2
    activity = y_train[:, remove_size:-remove_size]
    activity = np.sum(activity, axis=1)
    # Find pages with activity
    exists_activity_idx = np.where(activity > 0)[0]
    n_with_activity = exists_activity_idx.shape[0]
    print('Train pages with activity: %d (%1.2f %% of total)' % (n_with_activity, 100 * n_with_activity / n_train))

    if n_with_activity < n_train / 2:
        print('Balancing strategy: zero/exists activity')
        zero_activity_idx = np.where(activity == 0)[0]
        # Pages without any activity
        x_train_1 = x_train[zero_activity_idx]
        y_train_1 = y_train[zero_activity_idx]
        # Pages with activity
        x_train_2 = x_train[exists_activity_idx]
        y_train_2 = y_train[exists_activity_idx]
        # print('Pages without activity:', x_train_1.shape)
        # print('Pages with activity:', x_train_2.shape)
    else:
        print('Balancing strategy: low/high activity')
        sorted_idx = np.argsort(activity)
        low_activity_idx = sorted_idx[:int(n_train / 2)]
        high_activity_idx = sorted_idx[int(n_train / 2):]
        # Pages with low activity
        x_train_1 = x_train[low_activity_idx]
        y_train_1 = y_train[low_activity_idx]
        # Pages with high activity
        x_train_2 = x_train[high_activity_idx]
        y_train_2 = y_train[high_activity_idx]
        # print('Pages with low activity:', x_train_1.shape)
        # print('Pages with high activity:', x_train_2.shape)
    return x_train_1, y_train_1, x_train_2, y_train_2


if __name__ == '__main__':

    id_try_list = [i for i in range(4)]
    train_fraction = 0.75
    dataset_name = constants.MASS_SS_NAME

    batch_size = 32
    task_mode = constants.N2_RECORD
    which_expert = 1
    verbose = False

    params = pkeys.default_params.copy()
    print('\nChecking dataset %s_%s' % (dataset_name, task_mode))
    dataset = load_dataset(dataset_name, params=params)
    all_train_ids = dataset.train_ids
    all_considered_val_ids_list = []
    n_iters_per_epoch_list = []
    for id_try in id_try_list:
        print('\nValidation split %d' % id_try)
        # Generate split
        train_ids, val_ids = utils.split_ids_list_v2(all_train_ids, split_id=id_try, train_fraction=train_fraction)
        all_considered_val_ids_list.append(val_ids)
        print('Training set IDs (n=%d)' % len(train_ids))
        data_train = FeederDataset(dataset, train_ids, task_mode, which_expert=which_expert)
        print('Validation set IDs (n=%d):' % len(val_ids), val_ids)
        data_val = FeederDataset(dataset, val_ids, task_mode, which_expert=which_expert)
        # Check number of pages available
        x_train, y_train = data_train.get_data_for_training(
            border_size=0, forced_mark_separation_size=0, verbose=verbose)
        x_val, _ = data_val.get_data_for_training(
            border_size=0, forced_mark_separation_size=0, verbose=verbose)
        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        print("Pages: %d train, %d val." % (x_train.shape[0], x_val.shape[0]))
        x_train_1, y_train_1, x_train_2, y_train_2 = split_train(x_train, y_train, dataset.page_size, 0)
        n_smallest = min(x_train_1.shape[0], x_train_2.shape[0])
        n_iters_per_epoch = int(n_smallest / (batch_size / 2))
        n_iters_per_epoch_list.append(n_iters_per_epoch)
        print("With batch size %d, %d iterations before repeating pages (5 epochs = %d iters)" % (
            batch_size, n_iters_per_epoch, 5 * n_iters_per_epoch))

    print("\nOn average, %d iterations before repeating pages (5 epochs = %d iters)" % (
        np.mean(n_iters_per_epoch_list),
        5 * np.mean(n_iters_per_epoch_list)
    ))
    all_considered_val_ids_list = np.concatenate(all_considered_val_ids_list).flatten()
    print("Dataset alltrain size %d. Total val count %d. Total val unique count %d." % (
        len(dataset.train_ids),
        len(all_considered_val_ids_list),
        len(np.unique(all_considered_val_ids_list))
    ))
