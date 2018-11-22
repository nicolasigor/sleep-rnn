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
    dropout_rest_lstm_list = [None, constants.SEQUENCE_DROP]
    dropout_fc_list = [None, constants.SEQUENCE_DROP]
    use_log_list = [False, True]
    class_weights_list = [None, constants.BALANCED]

    # Create experiment
    parameters_list = list(itertools.product(
        dropout_rest_lstm_list,
        dropout_fc_list,
        use_log_list,
        class_weights_list))
    print('Number of combinations to be evaluated: %d' % len(parameters_list))

    for dropout_rest_lstm, dropout_fc, use_log, class_weights in parameters_list:
        experiment_dir = os.path.join(
            'results', 'grid_20181121',
            'dorest_%s_dofc_%s_log_%s_w_%s'
            % (dropout_rest_lstm, dropout_fc, use_log, class_weights)
        )
        print('This run directory: %s' % experiment_dir)

        # Load data
        dataset = MASS(load_checkpoint=True)

        # Update params
        params = param_keys.default_params
        params[param_keys.PAGE_DURATION] = dataset.page_duration
        params[param_keys.FS] = dataset.fs

        # Grid params
        params[param_keys.DROPOUT_REST_LSTM] = dropout_rest_lstm
        params[param_keys.DROPOUT_FC] = dropout_fc
        params[param_keys.USE_LOG] = use_log
        params[param_keys.CLASS_WEIGHTS] = class_weights

        # Create model
        model = WaveletBLSTM(params, logdir=experiment_dir)

        # Get training set ids
        print('Loading training set and splitting')
        all_train_ids = dataset.train_ids

        # Split to form validation set
        # TODO: quitar esta trampita
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
