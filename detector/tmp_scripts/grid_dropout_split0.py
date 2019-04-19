from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import os
import itertools
import datetime

import numpy as np

detector_path = '..'
results_folder = 'results'
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA
from sleep.mass_k import MASSK
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

    id_try_list = [0, 1]

    experiment_name = 'grid_dropout_whole_night'
    whole_night = True

    # Select database for training
    dataset_name_list = [constants.MASS_NAME]
    which_expert = 1

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    drop_rate_before_lstm_list = [0.0, 0.3, 0.5]
    drop_rate_hidden_list = [0.0, 0.5]
    drop_rate_output_list = [0.0, 0.2]

    # Create experiment
    parameters_list = list(itertools.product(
        drop_rate_before_lstm_list,
        drop_rate_hidden_list,
        drop_rate_output_list
    ))
    print('Number of combinations to be evaluated: %d' % len(parameters_list))

    for dataset_name in dataset_name_list:
        # Load data
        errors.check_valid_value(
            dataset_name, 'dataset_name',
            [constants.MASS_NAME, constants.INTA_NAME, constants.MASSK_NAME])
        if dataset_name == constants.MASS_NAME:
            dataset = MASS(load_checkpoint=True)
        elif dataset_name == constants.INTA_NAME:
            dataset = INTA(load_checkpoint=True)
        else:
            dataset = MASSK(load_checkpoint=True)

        # Update general params
        params = param_keys.default_params.copy()
        params[param_keys.PAGE_DURATION] = dataset.page_duration
        params[param_keys.FS] = dataset.fs

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
                which_expert=which_expert, verbose=True,
                whole_night=whole_night)
            x_val, y_val = dataset.get_subset_data(
                val_ids, augmented_page=False, border_size=border_size,
                which_expert=which_expert, verbose=True,
                whole_night=whole_night)

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

            for drop_rate_before_lstm, drop_rate_hidden, drop_rate_output in parameters_list:

                params[param_keys.DROP_RATE_BEFORE_LSTM] = drop_rate_before_lstm
                params[param_keys.DROP_RATE_HIDDEN] = drop_rate_hidden
                params[param_keys.DROP_RATE_OUTPUT] = drop_rate_output

                # Path to save results of run
                logdir = os.path.join(
                    results_folder,
                    '%s_train_%s' % (experiment_name, dataset_name),
                    'beforelstm_%s_hidden_%s_output_%s'
                    % (drop_rate_before_lstm, drop_rate_hidden, drop_rate_output),
                    'seed%d' % id_try
                )
                print('This run directory: %s' % logdir)

                # Create model
                model = WaveletBLSTM(params, logdir=logdir)

                # Train model
                model.fit(x_train, y_train, x_val, y_val)

                # ----- Obtain AF1 metric
                x_val_m, _ = dataset.get_subset_data(
                    val_ids, augmented_page=False, border_size=border_size,
                    which_expert=which_expert, verbose=False,
                    whole_night=whole_night)

                y_pred_val = []
                for i, sub_data in enumerate(x_val_m):
                    print('Val: Predicting ID %s' % val_ids[i])
                    this_pred = model.predict_proba(sub_data)
                    # Keep only probability of class one
                    this_pred = this_pred[..., 1]
                    y_pred_val.append(this_pred)

                _, y_val_m = dataset.get_subset_data(
                    val_ids, augmented_page=False, border_size=0,
                    which_expert=which_expert, verbose=False,
                    whole_night=whole_night)
                pages_val = dataset.get_subset_pages(
                    val_ids, verbose=False, whole_night=whole_night)

                val_af1 = metrics.average_f1_with_list(
                    y_val_m, y_pred_val, pages_val,
                    fs_real=dataset.fs, fs_predicted=dataset.fs // 8, thr=0.5,
                    postprocess_predicted=True,
                    min_duration=0.2, min_separation=0.3, max_duration=4)
                print('Validation AF1: %1.6f' % val_af1)

                metric_dict = {
                    'description':
                        'BSF grid dropout, whole_night %s'
                        % whole_night,
                    'val_seed': seed,
                    'database': dataset_name,
                    'val_af1': float(val_af1)
                }
                with open(os.path.join(model.logdir, 'metric.json'), 'w') as outfile:
                    json.dump(metric_dict, outfile)

                print('')
