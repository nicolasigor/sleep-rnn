from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from sleep.data.inta_ss import IntaSS
from sleep.data.mass_kc import MassKC
from sleep.data.mass_ss import MassSS
from sleep.data import postprocessing, data_ops, metrics
from sleep.neuralnet.models import WaveletBLSTM
from sleep.utils import pkeys
from sleep.utils import checks
from sleep.utils import constants

RESULTS_PATH = os.path.join(project_root, 'sleep', 'results')
SEED = 123


if __name__ == '__main__':

    # Select database for training
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    whole_night = True
    verbose = True
    # Description for metrics json
    description_str = 'demo'
    # Path to save results of run
    logdir_name = 'demo'

    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [
            constants.MASS_KC_NAME,
            constants.MASS_SS_NAME,
            constants.INTA_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=True)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(load_checkpoint=True)
    else:
        dataset = IntaSS(load_checkpoint=True)

    # Update params
    params = pkeys.default_params.copy()
    params[pkeys.PAGE_DURATION] = dataset.page_duration
    params[pkeys.FS] = dataset.fs

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    # Split to form validation set
    train_ids, val_ids = data_ops.split_ids_list(
        all_train_ids, seed=SEED)
    print('Training set IDs:', train_ids)
    print('Validation set IDs:', val_ids)

    # Get data
    border_size = params[pkeys.BORDER_DURATION] * params[pkeys.FS]
    x_train, y_train = dataset.get_subset_data(
        train_ids,
        augmented_page=True,
        border_size=border_size,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)
    x_val, y_val = dataset.get_subset_data(
        val_ids,
        augmented_page=True,
        border_size=border_size,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)

    # Transform to numpy arrays
    x_train_np = np.concatenate(x_train, axis=0)
    y_train_np = np.concatenate(y_train, axis=0)
    x_val_np = np.concatenate(x_val, axis=0)
    y_val_np = np.concatenate(y_val, axis=0)

    # Shuffle training set
    x_train_np, y_train_np = data_ops.shuffle_data(
        x_train_np, y_train_np, seed=SEED)

    print('Training set shape', x_train_np.shape, y_train_np.shape)
    print('Validation set shape', x_val_np.shape, y_val_np.shape)

    # Create model
    if whole_night:
        descriptor = '_whole_night_'
    else:
        descriptor = '_'
    logdir = os.path.join(
        RESULTS_PATH,
        '%s%strain_%s' % (logdir_name, descriptor, dataset_name))
    print('This run directory: %s' % logdir)

    model = WaveletBLSTM(params, logdir=logdir)

    # Train model
    model.fit(x_train_np, y_train_np, x_val_np, y_val_np)

    # Get metrics
    if dataset_name in [constants.MASS_SS_NAME, constants.INTA_SS_NAME]:
        min_separation = params[pkeys.SS_MIN_SEPARATION]
        min_duration = params[pkeys.SS_MIN_DURATION]
        max_duration = params[pkeys.SS_MAX_DURATION]
    else:
        min_separation = params[pkeys.KC_MIN_SEPARATION]
        min_duration = params[pkeys.KC_MIN_DURATION]
        max_duration = params[pkeys.KC_MAX_DURATION]

    # ----- Obtain AF1 metric
    x_val_m, _ = dataset.get_subset_data(
        val_ids,
        augmented_page=False,
        border_size=border_size,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)
    pages_val = dataset.get_subset_pages(
        val_ids,
        whole_night=whole_night,
        verbose=verbose)

    print('Predicting Validation set')
    y_pred_val = model.predict_proba_with_list(x_val_m, verbose=verbose)
    print('Done set')

    y_pred_val_stamps = postprocessing.generate_mark_intervals_with_list(
        y_pred_val,
        pages_val,
        fs_input=200 // 8,
        fs_output=200,
        thr=0.5,
        min_separation=min_separation,
        min_duration=min_duration,
        max_duration=max_duration)

    y_val_stamps = dataset.get_subset_stamps(
        val_ids,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)

    val_af1_at_half_thr = metrics.average_metric_with_list(
        y_val_stamps,
        y_pred_val_stamps,
        verbose=verbose)

    print('Validation AF1 with thr 0.5: %1.6f' % val_af1_at_half_thr)

    metric_dict = {
        'description': description_str,
        'val_seed': SEED,
        'database': dataset_name,
        'val_af1': float(val_af1_at_half_thr)
    }
    with open(os.path.join(model.logdir, 'metric.json'), 'w') as outfile:
        json.dump(metric_dict, outfile)

    # Measure inter-subject dispersion for training set

    x_train_m, _ = dataset.get_subset_data(
        train_ids,
        augmented_page=False,
        border_size=border_size,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)
    pages_train = dataset.get_subset_pages(
        train_ids,
        whole_night=whole_night,
        verbose=verbose)

    print('Predicting Training set')
    y_pred_train = model.predict_proba_with_list(x_train_m, verbose=verbose)
    print('Done set')

    y_pred_train_stamps = postprocessing.generate_mark_intervals_with_list(
        y_pred_train,
        pages_train,
        fs_input=200 // 8,
        fs_output=200,
        thr=0.5,
        min_separation=min_separation,
        min_duration=min_duration,
        max_duration=max_duration)

    y_train_stamps = dataset.get_subset_stamps(
        train_ids,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)

    be_stats = [
        metrics.by_event_confusion(this_y, this_y_pred, iou_thr=0.3)
        for (this_y, this_y_pred) in zip(y_train_stamps, y_pred_train_stamps)]

    recall_list = [this_stat[constants.RECALL] for this_stat in be_stats]
    precision_list = [this_stat[constants.PRECISION] for this_stat in be_stats]

    print('Train Recall %1.4f +- %1.4f'
          % (np.mean(recall_list), np.std(recall_list)))
    print('Train Precision %1.4f +- %1.4f'
          % (np.mean(precision_list), np.std(precision_list)))
    print('')
