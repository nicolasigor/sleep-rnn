from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np

detector_path = '..'
results_path = os.path.join(detector_path, 'results')
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA
from neuralnet.models import WaveletBLSTM
from evaluation import data_manipulation
from utils import param_keys
from utils import constants
from utils import errors

SEED = 123


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    # Select database for training
    dataset_name = constants.INTA_NAME
    which_expert = 1

    # Path to save results of run
    logdir = '20190328_v2bn_fixed_inta'
    logdir = os.path.join(
        results_path,
        '%s_train_%s' % (logdir, dataset_name),
        'bsf_try%d' % 1
    )

    # Load data
    errors.check_valid_value(
        dataset_name, 'dataset_name',
        [constants.MASS_NAME, constants.INTA_NAME])
    if dataset_name == constants.MASS_NAME:
        dataset = MASS(load_checkpoint=True)
    else:
        dataset = INTA(load_checkpoint=True)

    # Update params
    params = param_keys.default_params.copy()
    params[param_keys.PAGE_DURATION] = dataset.page_duration
    params[param_keys.FS] = dataset.fs
    # params[param_keys.MODEL_VERSION] = constants.V1
    # params[param_keys.MAX_ITERS] = 15000

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
        x_train, y_train, seed=SEED)

    print('Training set shape', x_train.shape, y_train.shape)
    print('Validation set shape', x_val.shape, y_val.shape)

    # Create model
    print('This run directory: %s' % logdir)
    model = WaveletBLSTM(params, logdir=logdir)

    # Train model
    model.fit(x_train, y_train, x_val, y_val)
