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
    clip_grad_clip_norm_list = [(True, 1), (True, 3), (False, 5)]
    type_optimizer_list = [constants.ADAM_OPTIMIZER, constants.SGD_OPTIMIZER]
    learning_rate_list = [0.1, 0.01, 0.001]
    type_loss_list = [constants.CROSS_ENTROPY_LOSS, constants.DICE_LOSS]

    # Create experiment
    parameters_list = list(itertools.product(
        clip_grad_clip_norm_list,
        type_optimizer_list,
        learning_rate_list,
        type_loss_list
    ))
    print('Number of combinations to be evaluated: %d' % len(parameters_list))

    for clip_grad_clip_norm, type_optimizer, learning_rate, type_loss in parameters_list:
        clip_grad = clip_grad_clip_norm[0]
        clip_norm = clip_grad_clip_norm[1]
        experiment_dir = os.path.join(
            'results', 'grid_20181122',
            'cgrad_%s_cnorm_%s_opt_%s_lr_%s_loss_%s'
            % (clip_grad, clip_norm, type_optimizer, learning_rate, type_loss)
        )
        print('This run directory: %s' % experiment_dir)

        # Load data
        dataset = MASS(load_checkpoint=True)

        # Update params
        params = param_keys.default_params
        params[param_keys.PAGE_DURATION] = dataset.page_duration
        params[param_keys.FS] = dataset.fs

        # Grid params
        params[param_keys.CLIP_GRADIENTS] = clip_grad
        params[param_keys.CLIP_NORM] = clip_norm
        params[param_keys.TYPE_OPTIMIZER] = type_optimizer
        params[param_keys.LEARNING_RATE] = learning_rate
        params[param_keys.TYPE_LOSS] = type_loss

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
