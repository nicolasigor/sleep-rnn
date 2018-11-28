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
    learning_rate_exp_list = [3]
    opt_momentum_list = [
        (constants.ADAM_OPTIMIZER, 0.0),  # Adam ignores momentum
        (constants.SGD_OPTIMIZER, 0.5),
        (constants.SGD_OPTIMIZER, 0.9),
        (constants.RMSPROP_OPTIMIZER, 0.0),
        (constants.RMSPROP_OPTIMIZER, 0.5),
        (constants.RMSPROP_OPTIMIZER, 0.9),
    ]
    batch_size_list = [32, 128]

    # Create experiment
    parameters_list = list(itertools.product(
        learning_rate_exp_list,
        opt_momentum_list,
        batch_size_list
    ))
    print('Number of combinations to be evaluated: %d' % len(parameters_list))

    for learning_rate, opt_momentum, batch_size in parameters_list:
        opt_name = opt_momentum[0]
        momentum = opt_momentum[1]

        experiment_dir = os.path.join(
            'results', 'grid_20181128',
            'lr_%d_batch_%d_opt_%s_m_%1.2f'
            % (learning_rate, batch_size, opt_name, momentum)
        )
        print('This run directory: %s' % experiment_dir)

        # Load data
        dataset = MASS(load_checkpoint=True)

        # Update params
        params = param_keys.default_params
        params[param_keys.PAGE_DURATION] = dataset.page_duration
        params[param_keys.FS] = dataset.fs

        params[param_keys.MAX_EPOCHS] = 100

        # Grid params
        params[param_keys.LEARNING_RATE] = 10**(-learning_rate)
        params[param_keys.BATCH_SIZE] = batch_size
        params[param_keys.TYPE_OPTIMIZER] = opt_name
        params[param_keys.MOMENTUM] = momentum

        # Create model
        model = WaveletBLSTM(params, logdir=experiment_dir)

        # Get training set ids
        print('Loading training set and splitting')
        all_train_ids = dataset.train_ids

        # Split to form validation set
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
