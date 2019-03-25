from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

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

    # Load data
    dataset = MASS(load_checkpoint=True)

    # Update params
    params = param_keys.default_params.copy()
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

    # Get data
    border_size = get_border_size(params)
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

    # Create model
    logdir = os.path.join('results', '20190325_fixed_loading', 'bsf_try3')
    print('This run directory: %s' % logdir)
    model = WaveletBLSTM(params, logdir=logdir)

    # Train model
    model.fit(x_train, y_train, x_val, y_val)
