"""

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

from sleep.inta import INTA
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

    # n_try = 0

    results_dir = os.path.join('..', 'results')
    # ckpt_folder = 'bsf_20190106/version_v2_typebn_bn_try_%d' % n_try
    ckpt_folder = 'v1_bn_fixed_files_trainINTA'

    ckpt_path = os.path.join(results_dir, ckpt_folder)

    # Load data
    dataset = INTA(load_checkpoint=True)

    # Restore params of ckpt
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        params = json.load(infile)
    # Overwrite defaults
    # params.update(param_keys.default_params)

    print('Restoring from %s' % ckpt_path)
    pprint.pprint(params)

    # filename = os.path.join(ckpt_path, 'bsf.json')
    # with open(filename, 'r') as infile:
    #     bsf_stats = json.load(infile)
    # print('BSF stats on validation set:')
    # print(bsf_stats)

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
    save_dir = os.path.join(results_dir, 'predictions_inta', ckpt_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving predictions at %s' % save_dir)
    np.save(os.path.join(save_dir, 'y_pred_train.npy'), y_pred_train)
    np.save(os.path.join(save_dir, 'y_pred_val.npy'), y_pred_val)
    np.save(os.path.join(save_dir, 'y_pred_test.npy'), y_pred_test)
    print('Predictions saved')

    # # ----- Augmented predictions
    # augmented_page = True
    # which_expert = 1
    #
    # x_train, y_train = dataset.get_subset_data(
    #     train_ids, augmented_page=augmented_page, border_size=border_size,
    #     which_expert=which_expert, verbose=True)
    # x_val, y_val = dataset.get_subset_data(
    #     val_ids, augmented_page=augmented_page, border_size=border_size,
    #     which_expert=which_expert, verbose=True)
    # x_test, y_test = dataset.get_subset_data(
    #     test_ids, augmented_page=augmented_page, border_size=border_size,
    #     which_expert=which_expert, verbose=True)
    #
    # # We keep each patient separate, to see variation of performance
    # # between individuals
    # y_pred_train = []
    # y_pred_val = []
    # y_pred_test = []
    #
    # # Start prediction
    # for i, sub_data in enumerate(x_train):
    #     print('Train: Predicting ID %s' % train_ids[i])
    #     this_pred = model.predict_proba_augmented(sub_data)
    #     y_pred_train.append(this_pred)
    # for i, sub_data in enumerate(x_val):
    #     print('Val: Predicting ID %s' % val_ids[i])
    #     this_pred = model.predict_proba_augmented(sub_data)
    #     y_pred_val.append(this_pred)
    # for i, sub_data in enumerate(x_test):
    #     print('Test: Predicting ID %s' % test_ids[i])
    #     this_pred = model.predict_proba_augmented(sub_data)
    #     y_pred_test.append(this_pred)
    #
    # # Save predictions
    # save_dir = os.path.join(results_dir, 'predictions_try_%d' % n_try,
    #                         ckpt_folder)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # print('Saving predictions at %s' % save_dir)
    # np.save(os.path.join(save_dir, 'y_pred_train_augmented.npy'), y_pred_train)
    # np.save(os.path.join(save_dir, 'y_pred_val_augmented.npy'), y_pred_val)
    # np.save(os.path.join(save_dir, 'y_pred_test_augmented.npy'), y_pred_test)
    # print('Predictions saved')
