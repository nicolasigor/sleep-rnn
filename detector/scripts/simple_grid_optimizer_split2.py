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


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    # Grid search
    trainable_wavelet_list = [True]
    drop_rate_list = [0.5]
    loss_opt_lr_m_batch_list = [
        (constants.CROSS_ENTROPY_LOSS, constants.ADAM_OPTIMIZER, 3, 0.0, 32),
        (constants.CROSS_ENTROPY_LOSS, constants.ADAM_OPTIMIZER, 4, 0.0, 32),
        (constants.CROSS_ENTROPY_LOSS, constants.SGD_OPTIMIZER, 3, 0.9, 32),
        (constants.CROSS_ENTROPY_LOSS, constants.SGD_OPTIMIZER, 4, 0.9, 32),
        (constants.DICE_LOSS, constants.ADAM_OPTIMIZER, 4, 0.0, 128),
        (constants.DICE_LOSS, constants.SGD_OPTIMIZER, 3, 0.5, 128),
        (constants.DICE_LOSS, constants.SGD_OPTIMIZER, 3, 0.9, 128)
    ]

    # Create experiment
    parameters_list = list(itertools.product(
        trainable_wavelet_list,
        drop_rate_list,
        loss_opt_lr_m_batch_list
    ))
    print('Number of combinations to be evaluated: %d' % len(parameters_list))

    # Load data
    dataset = MASS(load_checkpoint=True)

    # Update params
    params = param_keys.default_params
    params[param_keys.PAGE_DURATION] = dataset.page_duration
    params[param_keys.FS] = dataset.fs

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    # Split to form validation set
    train_ids, val_ids = data_manipulation.split_ids_list(
        all_train_ids, seed=SEED)
    print('Training set IDs:', train_ids)
    print('Validation set IDs:', val_ids)

    border_size = get_border_size(params)

    # Get data
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

    # Initiate grid search
    for trainable_wavelet, drop_rate, loss_opt_lr_m_batch in parameters_list:
        type_loss = loss_opt_lr_m_batch[0]
        type_opt = loss_opt_lr_m_batch[1]
        learning_rate = loss_opt_lr_m_batch[2]
        momentum = loss_opt_lr_m_batch[3]
        batch_size = loss_opt_lr_m_batch[4]

        experiment_dir = os.path.join(
            'results', 'grid_20181216',
            'loss_%s_opt_%s_lr_%d_m_%1.1f_batch_%d_trainwave_%d_drop_%1.1f'
            % (type_loss, type_opt, learning_rate, momentum, batch_size,
               int(trainable_wavelet), drop_rate)
        )
        print('This run directory: %s' % experiment_dir)

        # Grid params
        params[param_keys.TRAINABLE_WAVELET] = trainable_wavelet
        params[param_keys.DROP_RATE] = drop_rate
        params[param_keys.TYPE_LOSS] = type_loss
        params[param_keys.TYPE_OPTIMIZER] = type_opt
        params[param_keys.LEARNING_RATE] = 10 ** (-learning_rate)
        params[param_keys.MOMENTUM] = momentum
        params[param_keys.BATCH_SIZE] = batch_size

        # Create model
        model = WaveletBLSTM(params, logdir=experiment_dir)

        # Train model
        model.fit(x_train, y_train, x_val, y_val)
