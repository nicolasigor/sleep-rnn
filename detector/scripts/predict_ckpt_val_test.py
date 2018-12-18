from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json

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

    # ckpt_path = '../results/grid_20181216/loss_cross_entropy_loss_opt_adam_optimizer_lr_4_m_0.0_batch_32_trainwave_1_drop_0.3'
    ckpt_path = '../results/grid_20181217/loss_cross_entropy_loss_opt_adam_optimizer_lr_3_m_0.0_batch_32_trainwave_0_drop_0.3'

    # Restore params
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        params = json.load(infile)

    print('Restoring from %s' % ckpt_path)
    print(params)

    filename = os.path.join(ckpt_path, 'bsf.json')
    with open(filename, 'r') as infile:
        bsf_stats = json.load(infile)
    print('BSF stats on validation set:')
    print(bsf_stats)

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

    x_test, y_test = 0, 0

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
    model = WaveletBLSTM(params, logdir='results/demo_logs_graph')

    # Train model
    model.fit(x_train, y_train, x_val, y_val)
