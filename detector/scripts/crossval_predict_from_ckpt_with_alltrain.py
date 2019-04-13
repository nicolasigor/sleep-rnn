from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import pprint

import numpy as np

detector_path = '..'
results_path = os.path.join(detector_path, 'results')
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA
from sleep.mass_k import MASSK
from neuralnet.models import WaveletBLSTM
from evaluation import data_manipulation
from utils import param_keys
from utils import constants
from utils import errors


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    n_seeds = 4

    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = '20190413_bsf_kc_using_angle'
    grid_folder_list = ['bsf']

    # Select database for prediction
    dataset_name = 'massk'

    # Load data
    errors.check_valid_value(
        dataset_name, 'dataset_name',
        [constants.MASS_NAME, constants.INTA_NAME, 'massk'])
    if dataset_name == constants.MASS_NAME:
        dataset = MASS(load_checkpoint=True)
    elif dataset_name == constants.INTA_NAME:
        dataset = INTA(load_checkpoint=True)
    else:
        dataset = MASSK(load_checkpoint=True)
    # Get training set ids
    all_train_ids = dataset.train_ids

    # Test data
    test_ids = dataset.test_ids

    print('')
    for folder_name in grid_folder_list:
        print('\nGrid setting: %s' % folder_name)
        af1_list = []
        for k in range(n_seeds):
            print('')
            ckpt_path = os.path.abspath(os.path.join(
                results_path,
                '%s_train_%s' % (ckpt_folder, dataset_name),
                '%s' % folder_name,
                'seed%d' % k
            ))

            # Restore params of ckpt
            params = param_keys.default_params.copy()
            filename = os.path.join(ckpt_path, 'params.json')
            with open(filename, 'r') as infile:
                # Overwrite previous defaults with run's params
                params.update(json.load(infile))

            print('Restoring from %s' % ckpt_path)
            # pprint.pprint(params)

            # Restore seed
            filename = os.path.join(ckpt_path, 'metric.json')
            with open(filename, 'r') as infile:
                metric_dict = json.load(infile)
                this_seed = metric_dict['val_seed']
                print('Validation split seed: %d' % this_seed)
                this_af1 = metric_dict['val_af1']
                af1_list.append(this_af1)

            # Split to form validation set
            train_ids, val_ids = data_manipulation.split_ids_list(
                all_train_ids, seed=this_seed)

            print('Training set IDs:', train_ids)
            print('Validation set IDs:', val_ids)

            # Get data for predictions
            border_size = get_border_size(params)
            x_train, y_train = dataset.get_subset_data(
                train_ids, border_size=border_size, verbose=False)
            x_val, y_val = dataset.get_subset_data(
                val_ids, border_size=border_size, verbose=False)
            x_test, y_test = dataset.get_subset_data(
                test_ids, border_size=border_size, verbose=False)

            # All train
            x_alltrain, y_alltrain = dataset.get_subset_data(
                all_train_ids, border_size=border_size, verbose=False)

            # Create model
            model = WaveletBLSTM(params,
                                 logdir=os.path.join('results', 'demo_predict'))
            # Load checkpoint
            model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

            # We keep each patient separate, to see variation of performance
            # between individuals
            y_pred_train = []
            y_pred_val = []
            y_pred_test = []

            y_pred_alltrain = []

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

            for i, sub_data in enumerate(x_alltrain):
                print('AllTrain: Predicting ID %s' % all_train_ids[i])
                this_pred = model.predict_proba(sub_data)
                y_pred_alltrain.append(this_pred)

            # Save predictions
            save_dir = os.path.abspath(os.path.join(
                results_path,
                'predictions_%s' % dataset_name,
                '%s_train_%s' % (ckpt_folder, dataset_name),
                '%s' % folder_name,
                'seed%d' % k
            ))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(os.path.join(save_dir, 'y_pred_train.npy'), y_pred_train)
            np.save(os.path.join(save_dir, 'y_pred_val.npy'), y_pred_val)
            np.save(os.path.join(save_dir, 'y_pred_test.npy'), y_pred_test)
            np.save(os.path.join(save_dir, 'y_pred_alltrain.npy'), y_pred_alltrain)
            print('Predictions saved at %s' % save_dir)
        print('')
        mean_af1 = np.mean(af1_list)
        std_af1 = np.std(af1_list)
        print('Val-AF1 List:', af1_list)
        print('Mean: %1.4f' % mean_af1)
        print('Std: %1.4f' % std_af1)
