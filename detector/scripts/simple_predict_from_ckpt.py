from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import pprint

import numpy as np

detector_path = '..'
results_path = os.path.join(detector_path, 'results')
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.mass_k import MASSK
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

    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = '20190420_grid_dropout_whole_night_train_mass/beforelstm_0.3_hidden_0.5_output_0.0/seed0'
    whole_night = True
    personalized = True

    # Select database for prediction
    dataset_name = constants.MASS_NAME

    ckpt_path = os.path.join(results_path, ckpt_folder)

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

    # Restore params of ckpt
    params = param_keys.default_params.copy()
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        # Overwrite previous defaults with run's params
        params.update(json.load(infile))

    print('Restoring from %s' % ckpt_path)
    pprint.pprint(params)

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    # Split to form validation set
    train_ids, val_ids = data_manipulation.split_ids_list(
        all_train_ids, seed=SEED)
    print('Training set IDs:', train_ids)
    print('Validation set IDs:', val_ids)

    # Get test data
    print('Loading testing')
    test_ids = dataset.test_ids
    print('Testing set IDs:', test_ids)

    # Get data for predictions
    border_size = get_border_size(params)
    x_train, y_train = dataset.get_subset_data(
        train_ids, border_size=border_size, verbose=True,
        whole_night=whole_night)
    x_val, y_val = dataset.get_subset_data(
        val_ids, border_size=border_size, verbose=True,
        whole_night=whole_night)
    x_test, y_test = dataset.get_subset_data(
        test_ids, border_size=border_size, verbose=True,
        whole_night=whole_night)

    # Create model
    model = WaveletBLSTM(params, logdir=os.path.join('results', 'demo_predict'))
    # Load checkpoint
    model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

    # We keep each patient separate, to see variation of performance
    # between individuals
    y_pred_train = []
    y_pred_val = []
    y_pred_test = []

    # Start prediction
    for i, sub_data in enumerate(x_train):
        print('Train: Predicting ID %s' % train_ids[i])
        this_pred = model.predict_proba(sub_data, personalize=personalized)
        y_pred_train.append(this_pred)
    for i, sub_data in enumerate(x_val):
        print('Val: Predicting ID %s' % val_ids[i])
        this_pred = model.predict_proba(sub_data, personalize=personalized)
        y_pred_val.append(this_pred)
    for i, sub_data in enumerate(x_test):
        print('Test: Predicting ID %s' % test_ids[i])
        this_pred = model.predict_proba(sub_data, personalize=personalized)
        y_pred_test.append(this_pred)

    # Save predictions
    if personalized:
        prediction_folder = 'predictions_%s_personalized' % dataset_name
    else:
        prediction_folder = 'predictions_%s' % dataset_name

    save_dir = os.path.join(
        results_path,
        prediction_folder,
        ckpt_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving predictions at %s' % save_dir)

    if whole_night:
        descriptor = '_whole_night_'
    else:
        descriptor = '_'

    np.save(
        os.path.join(save_dir, 'y_pred%strain.npy' % descriptor),
        y_pred_train)
    np.save(
        os.path.join(save_dir, 'y_pred%sval.npy' % descriptor),
        y_pred_val)
    np.save(
        os.path.join(save_dir, 'y_pred%stest.npy' % descriptor),
        y_pred_test)
    print('Predictions saved')
