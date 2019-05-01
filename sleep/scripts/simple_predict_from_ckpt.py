from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys

import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from sleep.data.inta_ss import IntaSS
from sleep.data.mass_kc import MassKC
from sleep.data.mass_ss import MassSS
from sleep.data import postprocessing, data_ops, metrics
from sleep.neuralnet.models import WaveletBLSTM
from sleep.utils import pkeys
from sleep.utils import checks
from sleep.utils import constants

RESULTS_PATH = os.path.join(project_root, 'sleep', 'results')
SEED = 123


if __name__ == '__main__':

    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = 'demo_dice_train_mass'
    dataset_name = constants.MASS_SS_NAME
    whole_night = True
    verbose = True

    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [
            constants.MASS_KC_NAME,
            constants.MASS_SS_NAME,
            constants.INTA_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=True)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(load_checkpoint=True)
    else:
        dataset = IntaSS(load_checkpoint=True)

    ckpt_path = os.path.join(RESULTS_PATH, ckpt_folder)

    # Restore params of ckpt
    params = pkeys.default_params.copy()
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        # Overwrite previous defaults with run's params
        params.update(json.load(infile))

    print('Restoring from %s' % ckpt_path)
    pprint(params)

    # Get training set ids
    print('Loading training set and splitting')
    all_train_ids = dataset.train_ids

    # Split to form validation set
    train_ids, val_ids = data_ops.split_ids_list(
        all_train_ids, seed=SEED)
    print('Training set IDs:', train_ids)
    print('Validation set IDs:', val_ids)

    # Get test data
    print('Loading testing')
    test_ids = dataset.test_ids
    print('Testing set IDs:', test_ids)

    # Get data
    border_size = params[pkeys.BORDER_DURATION] * params[pkeys.FS]
    x_train, _ = dataset.get_subset_data(
        train_ids,
        border_size=border_size,
        whole_night=whole_night,
        verbose=verbose)
    x_val, _ = dataset.get_subset_data(
        val_ids,
        border_size=border_size,
        whole_night=whole_night,
        verbose=verbose)
    x_test, _ = dataset.get_subset_data(
        test_ids,
        border_size=border_size,
        whole_night=whole_night,
        verbose=verbose)

    # Create model
    model = WaveletBLSTM(params, logdir=os.path.join('results', 'demo_predict'))
    # Load checkpoint
    model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

    # We keep each patient separate, to see variation of performance
    # between individuals
    print('Predicting Training set')
    y_pred_train = model.predict_proba_with_list(x_train, verbose=verbose)
    print('Predicting Validation set')
    y_pred_val = model.predict_proba_with_list(x_val, verbose=verbose)
    print('Predicting Test set')
    y_pred_test = model.predict_proba_with_list(x_test, verbose=verbose)
    print('Done sets.')

    # Save predictions
    prediction_folder = 'predictions_%s' % dataset_name
    save_dir = os.path.join(
        RESULTS_PATH,
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
