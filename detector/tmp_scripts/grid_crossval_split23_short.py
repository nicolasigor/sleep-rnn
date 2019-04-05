from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import os

import numpy as np

detector_path = '..'
results_folder = 'results'
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA
from neuralnet.models import WaveletBLSTM
from evaluation import data_manipulation
from evaluation import metrics
from utils import param_keys
from utils import constants
from utils import errors

SEED_LIST = [123, 234, 345, 456]


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    id_try_list = [2, 3]

    # -----
    # Grid search
    lstm_size_list = [512]

    experiment_name = '20190405_lstm_size'

    print('Number of combinations to be evaluated: %d'
          % len(lstm_size_list))

    # Select database for training
    dataset_name = constants.MASS_NAME
    which_expert = 1

    # Load data
    errors.check_valid_value(
        dataset_name, 'dataset_name',
        [constants.MASS_NAME, constants.INTA_NAME])
    if dataset_name == constants.MASS_NAME:
        dataset = MASS(load_checkpoint=True)
    else:
        dataset = INTA(load_checkpoint=True)

    # Update general params
    params = param_keys.default_params.copy()
    params[param_keys.PAGE_DURATION] = dataset.page_duration
    params[param_keys.FS] = dataset.fs

    # Grid winners so far
    params[param_keys.TYPE_BATCHNORM] = constants.BN
    params[param_keys.MODEL_VERSION] = constants.V3

    # Shorter training time
    params[param_keys.MAX_ITERS] = 20000

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    for id_try in id_try_list:
        # Choose seed
        seed = SEED_LIST[id_try]
        print('\nUsing validation split seed %d' % seed)
        # Split to form validation set
        train_ids, val_ids = data_manipulation.split_ids_list(
            all_train_ids, seed=seed)
        print('Training set IDs:', train_ids)
        print('Validation set IDs:', val_ids)

        # Get data
        border_size = get_border_size(params)
        x_train, y_train = dataset.get_subset_data(
            train_ids, augmented_page=True, border_size=border_size,
            which_expert=which_expert, verbose=True)
        x_val, y_val = dataset.get_subset_data(
            val_ids, augmented_page=False, border_size=border_size,
            which_expert=which_expert, verbose=True)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        # Shuffle training set
        x_train, y_train = data_manipulation.shuffle_data(
            x_train, y_train, seed=seed)

        print('Training set shape', x_train.shape, y_train.shape)
        print('Validation set shape', x_val.shape, y_val.shape)

        # Start grid search
        for lstm_size in lstm_size_list:
            # Path to save results of run
            logdir = os.path.join(
                results_folder,
                '%s_train_%s' % (experiment_name, dataset_name),
                '%s' % lstm_size,
                'seed%d' % id_try
            )
            print('This run directory: %s' % logdir)

            # Grid params
            params[param_keys.INITIAL_LSTM_UNITS] = lstm_size

            # Create model
            model = WaveletBLSTM(params, logdir=logdir)

            # Train model
            model.fit(x_train, y_train, x_val, y_val)

            # ----- Obtain AF1 metric
            x_val_m, _ = dataset.get_subset_data(
                val_ids, augmented_page=False, border_size=border_size,
                which_expert=which_expert, verbose=False)

            y_pred_val = []
            for i, sub_data in enumerate(x_val_m):
                print('Val: Predicting ID %s' % val_ids[i])
                this_pred = model.predict_proba(sub_data)
                # Keep only probability of class one
                this_pred = this_pred[..., 1]
                y_pred_val.append(this_pred)

            _, y_val_m = dataset.get_subset_data(
                val_ids, augmented_page=False, border_size=0,
                which_expert=which_expert, verbose=False)
            pages_val = dataset.get_subset_pages(val_ids, verbose=False)

            val_af1 = metrics.average_f1_with_list(
                y_val_m, y_pred_val, pages_val,
                fs_real=dataset.fs, fs_predicted=dataset.fs // 8, thr=0.5)
            print('Validation AF1: %1.6f' % val_af1)

            metric_dict = {
                'description': 'using lstm size equal to %s' % lstm_size,
                'val_seed': seed,
                'database': dataset_name,
                'val_af1': float(val_af1)
            }
            with open(os.path.join(model.logdir, 'metric.json'), 'w') as outfile:
                json.dump(metric_dict, outfile)

            print('')
