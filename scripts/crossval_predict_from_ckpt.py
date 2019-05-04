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

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleep.data.inta_ss import IntaSS
from sleep.data.mass_kc import MassKC
from sleep.data.mass_ss import MassSS
from sleep.data import utils
from sleep.neuralnet.models import WaveletBLSTM
from sleep.common import constants
from sleep.common import checks
from sleep.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try_list = [0, 1, 2, 3]

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = '20190503_bsf'
    task_mode = constants.N2_RECORD
    dataset_name_list = [
        constants.MASS_SS_NAME,
        constants.MASS_KC_NAME
    ]
    which_expert = 1
    predict_with_augmented_page = True
    verbose = False
    grid_folder_list = None
    # -----

    checks.check_valid_value(
        task_mode, 'task_mode',
        [constants.WN_RECORD, constants.N2_RECORD])

    for dataset_name in dataset_name_list:

        print('\nModel predicting on %s_%s' % (dataset_name, task_mode))

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
                    '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name)
                ))
            print('Grid settings found:')
            pprint(grid_folder_list)

        print('')
        for folder_name in grid_folder_list:
            print('\nGrid setting: %s' % folder_name)
            af1_list = []
            for k in id_try_list:
                print('')
                ckpt_path = os.path.abspath(os.path.join(
                    RESULTS_PATH,
                    '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name),
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
                train_ids, val_ids = utils.split_ids_list(
                    all_train_ids, seed=this_seed)

                print('Training set IDs:', train_ids)
                print('Validation set IDs:', val_ids)

                # Create model
                model = WaveletBLSTM(
                    params,
                    logdir=os.path.join('results', 'demo_predict'))
                # Load checkpoint
                model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

                # Get data for predictions
                border_size = params[pkeys.BORDER_DURATION] * params[pkeys.FS]

                ids_dict = {
                    constants.TRAIN_SUBSET: train_ids,
                    constants.VAL_SUBSET: val_ids,
                    constants.TEST_SUBSET: test_ids
                }

                # Save predictions
                save_dir = os.path.abspath(os.path.join(
                    RESULTS_PATH,
                    'predictions_%s' % dataset_name,
                    '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name),
                    '%s' % folder_name,
                    'seed%d' % k
                ))
                checks.ensure_directory(save_dir)

                # If we need to predict over N2, then we predict over whole
                # night and then, after postprocessing, we keep N2 pages only

                for key in ids_dict.keys():
                    x, _ = dataset.get_subset_data(
                        ids_dict[key],
                        augmented_page=predict_with_augmented_page,
                        border_size=border_size,
                        which_expert=which_expert,
                        pages_subset=constants.WN_RECORD,
                        normalize_clip=True,
                        normalization_mode=task_mode,
                        verbose=verbose)

                    # We keep each patient separate, to see variation of
                    # performance between individuals
                    print('Predicting %s' % key, flush=True)
                    y_pred = model.predict_proba_with_list(
                        x,
                        verbose=verbose,
                        with_augmented_page=predict_with_augmented_page)

                    np.save(
                        os.path.join(
                            save_dir, 'y_pred_%s_%s.npy' % (task_mode, key)),
                        y_pred)

                print('Predictions saved at %s' % save_dir)
            print('')
            mean_af1 = np.mean(af1_list)
            std_af1 = np.std(af1_list)
            print('Val-AF1 List:', af1_list)
            print('Mean: %1.4f' % mean_af1)
            print('Std: %1.4f' % std_af1)
