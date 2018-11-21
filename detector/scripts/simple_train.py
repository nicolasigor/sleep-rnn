from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

detector_path = '..'
sys.path.append(detector_path)

from sleep.mass import MASS
from neuralnet.models import WaveletBLSTM
from evaluation import data_manipulation
from utils import param_keys

SEED = 1234

if __name__ == '__main__':

    # Load data
    dataset = MASS(load_checkpoint=True)

    # Update params
    params = param_keys.default_params
    params[param_keys.PAGE_DURATION] = dataset.page_duration
    params[param_keys.FS] = dataset.fs

    # Create model
    model = WaveletBLSTM(params, logdir='results/demo_logs')

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    # Split to form validation set
    # TODO: quitar esta trampita
    train_ids, _ = data_manipulation.split_ids_list(
        all_train_ids, seed=SEED)
    val_ids = dataset.test_ids
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
