from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import os

import numpy as np

import sleep.data.data_ops

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
results_folder = 'results'
sys.path.append(project_root)

from sleep.data.old_mass_ss import MassSS
from sleep.data.old_inta_ss import IntaSS
from sleep.data import postprocessing, data_manipulation, metrics
from sleep.neuralnet.models import WaveletBLSTM
from sleep.utils import param_keys
from sleep.utils import constants
from sleep.utils import checks

SEED = 123


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    # Select database for training
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    whole_night = True

    # Path to save results of run
    logdir = 'demo'
    logdir = os.path.join(
        results_folder,
        '%s_train_%s' % (logdir, dataset_name))

    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [constants.MASS_SS_NAME, constants.INTA_SS_NAME])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=True)
    else:
        dataset = IntaSS(load_checkpoint=True)

    # Update params
    params = param_keys.default_params.copy()
    params[param_keys.PAGE_DURATION] = dataset.page_duration
    params[param_keys.FS] = dataset.fs

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    # Split to form validation set
    train_ids, val_ids = sleep.data.data_ops.split_ids_list(
        all_train_ids, seed=SEED)
    print('Training set IDs:', train_ids)
    print('Validation set IDs:', val_ids)

    # Get data
    border_size = get_border_size(params)
    x_train, y_train = dataset.get_subset_data(
        train_ids, augmented_page=True, border_size=border_size,
        which_expert=which_expert, verbose=True, whole_night=whole_night)
    x_val, y_val = dataset.get_subset_data(
        val_ids, augmented_page=False, border_size=border_size,
        which_expert=which_expert, verbose=True, whole_night=whole_night)

    # Transform to numpy arrays
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Shuffle training set
    x_train, y_train = sleep.data.data_ops.shuffle_data(
        x_train, y_train, seed=SEED)

    print('Training set shape', x_train.shape, y_train.shape)
    print('Validation set shape', x_val.shape, y_val.shape)

    # Create model
    print('This run directory: %s' % logdir)
    model = WaveletBLSTM(params, logdir=logdir)

    # Train model
    model.fit(x_train, y_train, x_val, y_val)

    # ----- Obtain AF1 metric
    x_val_m, _ = dataset.get_subset_data(
        val_ids, augmented_page=False, border_size=border_size,
        which_expert=which_expert, verbose=False, whole_night=whole_night)

    y_pred_val = []
    for i, sub_data in enumerate(x_val_m):
        print('Val: Predicting ID %s' % val_ids[i])
        this_pred = model.predict_proba(sub_data)
        # Keep only probability of class one
        this_pred = this_pred[..., 1]
        y_pred_val.append(this_pred)

    _, y_val_m = dataset.get_subset_data(
        val_ids, augmented_page=False, border_size=0,
        which_expert=which_expert, verbose=False, whole_night=whole_night)
    pages_val = dataset.get_subset_pages(val_ids, verbose=False,
                                         whole_night=whole_night)

    val_af1 = metrics.average_f1_with_list(
        y_val_m, y_pred_val, pages_val,
        fs_real=dataset.fs, fs_predicted=dataset.fs//8, thr=0.5)
    print('Validation AF1: %1.6f' % val_af1)

    metric_dict = {
        'description': 'demo',
        'val_seed': SEED,
        'database': dataset_name,
        'val_af1': float(val_af1)
    }
    with open(os.path.join(model.logdir, 'metric.json'), 'w') as outfile:
        json.dump(metric_dict, outfile)

    # Precision and recall for training set

    x_train_m, _ = dataset.get_subset_data(
        train_ids, augmented_page=False, border_size=border_size,
        which_expert=which_expert, verbose=False, whole_night=whole_night)

    y_pred_train = []
    for i, sub_data in enumerate(x_train_m):
        print('Train: Predicting ID %s' % train_ids[i])
        this_pred = model.predict_proba(sub_data)
        # Keep only probability of class one
        this_pred = this_pred[..., 1]
        y_pred_train.append(this_pred)

    _, y_train_m = dataset.get_subset_data(
        train_ids, augmented_page=False, border_size=0,
        which_expert=which_expert, verbose=False, whole_night=whole_night)
    pages_train = dataset.get_subset_pages(train_ids, verbose=False,
                                           whole_night=whole_night)

    y_pred_thr = postprocessing.generate_mark_intervals_with_list(
        y_pred_train, pages_train, 200 // 8, 200, thr=0.5,
        min_separation=0.3, min_duration=0.2,
        max_duration=4)
    y_stamps = postprocessing.generate_mark_intervals_with_list(
        y_train_m, pages_train, 200, 200, thr=None, postprocess=False)
    be_stats = [
        metrics.by_event_confusion(this_y, this_y_pred, iou_thr=0.3)
        for (this_y, this_y_pred) in zip(y_stamps, y_pred_thr)]
    for this_stat in be_stats:
        print('Recall, Precision:', this_stat['recall'], this_stat['precision'])
    print('')