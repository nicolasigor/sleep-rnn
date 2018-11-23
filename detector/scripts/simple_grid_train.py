from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import sys

import numpy as np

detector_path = '..'
sys.path.append(detector_path)

from sleep.mass import MASS
from neuralnet.models import WaveletBLSTM
from evaluation import data_manipulation
from utils import param_keys
from utils import constants

SEED = 123


if __name__ == '__main__':

    # Grid search
    # dropout_rest_lstm_list = [None, constants.SEQUENCE_DROP]
    # dropout_fc_list = [None, constants.SEQUENCE_DROP]
    # use_log_list = [False, True]
    # class_weights_list = [None, constants.BALANCED]
    # clip_norm_list = [0.25, 0.5, 1]
    learning_rate_exp_list = [3, 4, 5]
    initial_lstm_units_list = [64, 128]
    n_time_levels_list = [1, 2, 3]

    # Create experiment
    parameters_list = list(itertools.product(
        learning_rate_exp_list,
        initial_lstm_units_list,
        n_time_levels_list
    ))
    print('Number of combinations to be evaluated: %d' % len(parameters_list))

    for learning_rate, initial_lstm_units, n_time_levels in parameters_list:
        experiment_dir = os.path.join(
            'results', 'grid_20181123',
            'lr_%s_lstm_%s_ntime_%s'
            % (learning_rate, initial_lstm_units, n_time_levels)
        )
        print('This run directory: %s' % experiment_dir)

        # Load data
        dataset = MASS(load_checkpoint=True)

        # Update params
        params = param_keys.default_params
        params[param_keys.PAGE_DURATION] = dataset.page_duration
        params[param_keys.FS] = dataset.fs

        # Grid params
        params[param_keys.LEARNING_RATE] = 10**(-learning_rate)
        params[param_keys.INITIAL_LSTM_UNITS] = initial_lstm_units
        params[param_keys.N_TIME_LEVELS] = n_time_levels

        # Create model
        model = WaveletBLSTM(params, logdir=experiment_dir)

        # Get training set ids
        print('Loading training set and splitting')
        all_train_ids = dataset.train_ids

        # Split to form validation set
        # train_ids, _ = data_manipulation.split_ids_list(
        #     all_train_ids, seed=SEED)
        # val_ids = dataset.test_ids
        train_ids, val_ids = data_manipulation.split_ids_list(
             all_train_ids, seed=SEED)
        print('Training set IDs:', train_ids)
        print('Validation set IDs:', val_ids)

        # Get data
        border_size = model.get_border_size()
        x_train, y_train = dataset.get_subset_data(
            train_ids, augmented_page=True, border_size=border_size,
            which_expert=1, verbose=True)
        x_val, y_val = dataset.get_subset_data(
            val_ids, augmented_page=False, border_size=border_size,
            which_expert=1, verbose=True)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        # Shuffle training set
        x_train, y_train = data_manipulation.shuffle_data(
            x_train, y_train, seed=SEED)

        print('Training set shape', x_train.shape, y_train.shape)
        print('Validation set shape', x_val.shape, y_val.shape)

        # Train model
        model.fit(x_train, y_train, x_val, y_val)
