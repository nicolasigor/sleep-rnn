"""
Train: Predicting ID 9
Train: Predicting ID 1
Train: Predicting ID 14
Train: Predicting ID 10
Train: Predicting ID 17
Train: Predicting ID 7
Train: Predicting ID 3
Train: Predicting ID 11
Val: Predicting ID 19
Val: Predicting ID 5
Test: Predicting ID 2
Test: Predicting ID 6
Test: Predicting ID 12
Test: Predicting ID 13
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import pprint

import numpy as np

detector_path = '..'
sys.path.append(detector_path)

from sleep.mass import MASS
from neuralnet.models import WaveletBLSTM
from evaluation import data_manipulation
from utils import param_keys

SEED = 123


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    results_dir = os.path.join('..', 'results')
    # ckpt_folder = 'grid_20181216/loss_cross_entropy_loss_opt_adam_optimizer_lr_4_m_0.0_batch_32_trainwave_1_drop_0.3'
    ckpt_folder = 'grid_20181217/loss_cross_entropy_loss_opt_adam_optimizer_lr_3_m_0.0_batch_32_trainwave_0_drop_0.3'

    ckpt_path = os.path.join(results_dir, ckpt_folder)

    # Load data
    dataset = MASS(load_checkpoint=True)

    # Restore params of ckpt
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        params = json.load(infile)
    # Overwrite defaults
    params.update(param_keys.default_params)

    print('Restoring from %s' % ckpt_path)
    pprint.pprint(params)

    filename = os.path.join(ckpt_path, 'bsf.json')
    with open(filename, 'r') as infile:
        bsf_stats = json.load(infile)
    print('BSF stats on validation set:')
    print(bsf_stats)

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids
    # Split to form validation set
    train_ids, val_ids = data_manipulation.split_ids_list(
        all_train_ids, seed=SEED)
    print('Training set IDs:', train_ids)
    print('Validation set IDs:', val_ids)

    # Get test data
    print('Loading testing')
    test_ids = dataset.test_ids
    print('Testing set IDs:', test_ids)

    # Get data for predictions
    augmented_page = False
    which_expert = 1

    border_size = get_border_size(params)
    x_train, y_train = dataset.get_subset_data(
        train_ids, augmented_page=augmented_page, border_size=border_size,
        which_expert=which_expert, verbose=True)
    x_val, y_val = dataset.get_subset_data(
        val_ids, augmented_page=augmented_page, border_size=border_size,
        which_expert=which_expert, verbose=True)
    x_test, y_test = dataset.get_subset_data(
        test_ids, augmented_page=augmented_page, border_size=border_size,
        which_expert=which_expert, verbose=True)

    # Create model
    model = WaveletBLSTM(params, logdir='results/demo_predict')
    # Load checkpoint
    model.load_checkpoint(os.path.join(ckpt_path, 'model/ckpt'))

    # We keep each patient separate, to see variation of performance
    # between individuals
    y_pred_train = []
    y_pred_val = []
    y_pred_test = []

    # Start prediction
    for i, sub_data in enumerate(x_train):
        print('Train: Predicting ID %s' % train_ids[i])
        this_pred = model.predict_proba(sub_data)
        y_pred_train.append(this_pred)
    for i, sub_data in enumerate(x_val):
        print('Val: Predicting ID %s' % val_ids[i])
        this_pred = model.predict_proba(sub_data)
        y_pred_val.append(this_pred)
    for i, sub_data in enumerate(x_test):
        print('Test: Predicting ID %s' % test_ids[i])
        this_pred = model.predict_proba(sub_data)
        y_pred_test.append(this_pred)

    # Save predictions
    save_dir = os.path.join(results_dir, 'predictions', ckpt_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving predictions at %s' % save_dir)
    np.save(os.path.join(save_dir, 'y_pred_train.npy'), y_pred_train)
    np.save(os.path.join(save_dir, 'y_pred_val.npy'), y_pred_val)
    np.save(os.path.join(save_dir, 'y_pred_test.npy'), y_pred_test)
    print('Predictions saved')
