from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
results_folder = 'results'
sys.path.append(project_root)

from sleep.data.inta_ss import IntaSS
from sleep.data.mass_kc import MassKC
from sleep.data.mass_ss import MassSS
from sleep.data import data_ops
from sleep.neuralnet.models import WaveletBLSTM
from sleep.utils import constants
from sleep.utils import checks
from sleep.utils import pkeys

RESULTS_PATH = os.path.join(project_root, 'sleep', 'results')


if __name__ == '__main__':

    seed_list = [0, 1, 2, 3]

    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = '20190502_bsf_norm_activity'
    whole_night = True
    # Select database for prediction
    dataset_name_list = [
        constants.MASS_SS_NAME,
        constants.MASS_KC_NAME
    ]

    with_augmented_page = True
    debug_force_n2stats = False
    debug_force_activitystats = True
    which_expert = 1
    grid_folder_list = None
    verbose = True

    if whole_night:
        descriptor = '_whole_night_'
    else:
        descriptor = '_'

    for dataset_name in dataset_name_list:

        print('\nModel predicting on %s%s' % (dataset_name, descriptor))

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

        # Get training set ids
        all_train_ids = dataset.train_ids

        # Test data
        test_ids = dataset.test_ids

        if grid_folder_list is None:

            grid_folder_list = os.listdir(os.path.join(
                    RESULTS_PATH,
                    '%s%strain_%s' % (ckpt_folder, descriptor, dataset_name)
                ))
            print('Grid settings found:')
            pprint(grid_folder_list)

        print('')
        for folder_name in grid_folder_list:
            print('\nGrid setting: %s' % folder_name)
            af1_list = []
            for k in seed_list:
                print('')
                ckpt_path = os.path.abspath(os.path.join(
                    RESULTS_PATH,
                    '%s%strain_%s' % (ckpt_folder, descriptor, dataset_name),
                    '%s' % folder_name,
                    'seed%d' % k
                ))

                # Restore params of ckpt
                params = pkeys.default_params.copy()
                filename = os.path.join(ckpt_path, 'params.json')
                with open(filename, 'r') as infile:
                    # Overwrite previous defaults with run's params
                    params.update(json.load(infile))

                print('Restoring from %s' % ckpt_path)

                # Restore seed
                filename = os.path.join(ckpt_path, 'metric.json')
                with open(filename, 'r') as infile:
                    metric_dict = json.load(infile)
                    this_seed = metric_dict['val_seed']
                    print('Validation split seed: %d' % this_seed)
                    this_af1 = metric_dict['val_af1']
                    af1_list.append(this_af1)

                # Split to form validation set
                train_ids, val_ids = data_ops.split_ids_list(
                    all_train_ids, seed=this_seed)

                print('Training set IDs:', train_ids)
                print('Validation set IDs:', val_ids)

                # Get data for predictions
                border_size = params[pkeys.BORDER_DURATION] * params[pkeys.FS]

                # If we need to predict over N2, then we predict over whole
                # night but forcing N2 normalization to keep the same
                # normalization used for training
                if not whole_night:
                    debug_force_n2stats = True

                x_train, _ = dataset.get_subset_data(
                    train_ids,
                    border_size=border_size,
                    augmented_page=with_augmented_page,
                    which_expert=which_expert,
                    whole_night=True,
                    debug_force_n2stats=debug_force_n2stats,
                    debug_force_activitystats=debug_force_activitystats,
                    verbose=verbose)
                x_val, _ = dataset.get_subset_data(
                    val_ids,
                    border_size=border_size,
                    augmented_page=with_augmented_page,
                    which_expert=which_expert,
                    whole_night=True,
                    debug_force_n2stats=debug_force_n2stats,
                    debug_force_activitystats=debug_force_activitystats,
                    verbose=verbose)
                x_test, _ = dataset.get_subset_data(
                    test_ids,
                    border_size=border_size,
                    augmented_page=with_augmented_page,
                    which_expert=which_expert,
                    whole_night=True,
                    debug_force_n2stats=debug_force_n2stats,
                    debug_force_activitystats=debug_force_activitystats,
                    verbose=verbose)

                # Create model
                model = WaveletBLSTM(
                    params,
                    logdir=os.path.join('results', 'demo_predict'))

                # Load checkpoint
                model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

                # We keep each patient separate, to see variation of performance
                # between individuals
                print('Predicting Train', flush=True)
                y_pred_train = model.predict_proba_with_list(
                    x_train, verbose=verbose,
                    with_augmented_page=with_augmented_page)
                print('Predicting Val', flush=True)
                y_pred_val = model.predict_proba_with_list(
                    x_val, verbose=verbose,
                    with_augmented_page=with_augmented_page)
                print('Predicting Test', flush=True)
                y_pred_test = model.predict_proba_with_list(
                    x_test, verbose=verbose,
                    with_augmented_page=with_augmented_page)

                # Save predictions
                save_dir = os.path.abspath(os.path.join(
                    RESULTS_PATH,
                    'predictions_%s' % dataset_name,
                    '%s%strain_%s' % (ckpt_folder, descriptor, dataset_name),
                    '%s' % folder_name,
                    'seed%d' % k
                ))

                checks.ensure_directory(save_dir)

                np.save(
                    os.path.join(save_dir, 'y_pred%strain.npy' % descriptor),
                    y_pred_train)
                np.save(
                    os.path.join(save_dir, 'y_pred%sval.npy' % descriptor),
                    y_pred_val)
                np.save(
                    os.path.join(save_dir, 'y_pred%stest.npy' % descriptor),
                    y_pred_test)
                print('Predictions saved at %s' % save_dir)
            print('')
            mean_af1 = np.mean(af1_list)
            std_af1 = np.std(af1_list)
            print('Val-AF1 List:', af1_list)
            print('Mean: %1.4f' % mean_af1)
            print('Std: %1.4f' % std_af1)
