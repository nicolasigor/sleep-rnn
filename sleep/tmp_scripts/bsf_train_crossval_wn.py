from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
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
from sleep.data import data_ops, metrics, postprocessing
from sleep.neuralnet.models import WaveletBLSTM
from sleep.utils import constants
from sleep.utils import checks
from sleep.utils import pkeys

RESULTS_PATH = os.path.join(project_root, 'sleep', 'results')
SEED_LIST = [123, 234, 345, 456]


if __name__ == '__main__':

    id_try_list = [0, 1, 2, 3]

    # ----- Experiment settings
    experiment_name = 'bsf'
    task_mode = constants.WN_RECORD
    dataset_name_list = [
        constants.MASS_SS_NAME,
        constants.MASS_KC_NAME
    ]
    description_str = 'bsf'
    which_expert = 1
    predict_with_augmented_page = True
    verbose = False
    # -----

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    checks.check_valid_value(
        task_mode, 'task_mode',
        [constants.WN_RECORD, constants.N2_RECORD])

    for dataset_name in dataset_name_list:

        print('\nModel training on %s_%s' % (dataset_name, task_mode))

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

        # Update general params
        params = pkeys.default_params.copy()
        params[pkeys.PAGE_DURATION] = dataset.page_duration
        params[pkeys.FS] = dataset.fs

        # Get training set ids
        print('Loading training set and splitting')
        all_train_ids = dataset.train_ids

        # Parameters for postprocessing
        if dataset_name in [constants.MASS_SS_NAME, constants.INTA_SS_NAME]:
            min_separation = params[pkeys.SS_MIN_SEPARATION]
            min_duration = params[pkeys.SS_MIN_DURATION]
            max_duration = params[pkeys.SS_MAX_DURATION]
        else:
            min_separation = params[pkeys.KC_MIN_SEPARATION]
            min_duration = params[pkeys.KC_MIN_DURATION]
            max_duration = params[pkeys.KC_MAX_DURATION]

        for id_try in id_try_list:
            # Choose seed
            seed = SEED_LIST[id_try]
            print('\nUsing validation split seed %d' % seed)
            # Split to form validation set
            train_ids, val_ids = data_ops.split_ids_list(
                all_train_ids, seed=seed)
            print('Training set IDs:', train_ids)
            print('Validation set IDs:', val_ids)

            # Get data for training
            border_size = params[pkeys.BORDER_DURATION] * params[pkeys.FS]
            x_train, y_train = dataset.get_subset_data(
                train_ids,
                augmented_page=True,
                border_size=border_size,
                which_expert=which_expert,
                pages_subset=task_mode,
                normalize_clip=True,
                normalization_mode=task_mode,
                verbose=verbose)
            x_val, y_val = dataset.get_subset_data(
                val_ids,
                augmented_page=True,
                border_size=border_size,
                which_expert=which_expert,
                pages_subset=task_mode,
                normalize_clip=True,
                normalization_mode=task_mode,
                verbose=verbose)

            # Transform to numpy arrays
            x_train_np = np.concatenate(x_train, axis=0)
            y_train_np = np.concatenate(y_train, axis=0)
            x_val_np = np.concatenate(x_val, axis=0)
            y_val_np = np.concatenate(y_val, axis=0)

            # Shuffle training set
            x_train_np, y_train_np = data_ops.shuffle_data(
                x_train_np, y_train_np, seed=seed)

            print('Training set shape', x_train_np.shape, y_train_np.shape)
            print('Validation set shape', x_val_np.shape, y_val_np.shape)

            # Path to save results of run
            logdir = os.path.join(
                RESULTS_PATH,
                '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                'bsf',
                'seed%d' % id_try
            )
            print('This run directory: %s' % logdir)

            # Create model
            model = WaveletBLSTM(params, logdir=logdir)

            # Train model
            model.fit(x_train_np, y_train_np, x_val_np, y_val_np)

            # ----- Obtain AF1 metric
            x_val_m, _ = dataset.get_subset_data(
                val_ids,
                augmented_page=predict_with_augmented_page,
                border_size=border_size,
                which_expert=which_expert,
                pages_subset=constants.WN_RECORD,
                normalize_clip=True,
                normalization_mode=task_mode,
                verbose=verbose)

            print('Predicting Validation set')
            y_pred_val = model.predict_proba_with_list(
                x_val_m,
                verbose=verbose,
                with_augmented_page=predict_with_augmented_page)
            print('Done set')

            # Postprocessing
            wn_pages_val = dataset.get_subset_pages(
                val_ids,
                pages_subset=constants.WN_RECORD,
                verbose=verbose)
            y_pred_val_stamps = postprocessing.generate_mark_intervals_with_list(
                y_pred_val,
                wn_pages_val,
                fs_input=200 // 8,
                fs_output=200,
                thr=0.5,
                min_separation=min_separation,
                min_duration=min_duration,
                max_duration=max_duration)

            if task_mode == constants.N2_RECORD:
                # Keep only N2 stamps
                n2_pages_val = dataset.get_subset_pages(
                    val_ids,
                    pages_subset=constants.N2_RECORD,
                    verbose=verbose)
                y_pred_val_stamps = [data_ops.extract_pages_with_stamps(
                    this_y_pred_stamps, this_pages, dataset.page_size)
                    for (this_y_pred_stamps, this_pages)
                    in zip(y_pred_val_stamps, n2_pages_val)]

            y_val_stamps = dataset.get_subset_stamps(
                val_ids,
                which_expert=which_expert,
                pages_subset=task_mode,
                verbose=verbose)

            val_af1_at_half_thr = metrics.average_metric_with_list(
                y_val_stamps,
                y_pred_val_stamps,
                verbose=verbose)

            print('Validation AF1 with thr 0.5: %1.6f' % val_af1_at_half_thr)

            metric_dict = {
                'description': description_str,
                'val_seed': seed,
                'database': dataset_name,
                'task_mode': task_mode,
                'val_af1': float(val_af1_at_half_thr)
            }
            with open(os.path.join(model.logdir, 'metric.json'),
                      'w') as outfile:
                json.dump(metric_dict, outfile)

            print('')
