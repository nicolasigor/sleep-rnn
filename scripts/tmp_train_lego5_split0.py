from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import itertools
import json
import os
import pickle
from pprint import pprint
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    n_folds = 5
    fold_id_list = [0, 1, 2]  # [i for i in range(n_folds)]
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1

    # ----- Experiment settings
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = 'lego5_%dfold-cv_exp%d' % (n_folds, which_expert)
    task_mode = constants.N2_RECORD
    description_str = 'experiments'
    verbose = True
    # Complement experiment folder name with date
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_version_list = [
        constants.V43
    ]
    context_part_list = [
        'lstm_skip_early',
        'lstm_skip_late',
        'lstm'
    ]
    cv_seed_list = [
        0,
        1,
        2
    ]
    params_list = list(itertools.product(
        model_version_list, context_part_list
    ))

    # Common parameters
    base_params = pkeys.default_params.copy()
    base_params[pkeys.BORDER_DURATION] = 3
    # Conv part parameters: multi-d8 with or without residuals
    base_params[pkeys.BIGGER_STEM_FILTERS] = 64
    base_params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE] = 3
    base_params[pkeys.BIGGER_MAX_DILATION] = 8
    base_params[pkeys.BIGGER_STAGE_1_SIZE] = 1
    base_params[pkeys.BIGGER_STAGE_2_SIZE] = 1
    base_params[pkeys.BIGGER_STAGE_3_SIZE] = 0
    # Fitted params
    base_params[pkeys.BIGGER_CONVOLUTION_PART_OPTION] = 'multi_dilated'
    base_params[pkeys.BIGGER_STEM_KERNEL_SIZE] = 3
    base_params[pkeys.BIGGER_STEM_DEPTH] = 2
    base_params[pkeys.BIGGER_MULTI_TRANSFORMATION_BEFORE_ADD] = None
    # Context part parameters: lstm
    base_params[pkeys.BIGGER_LSTM_1_SIZE] = 256
    base_params[pkeys.BIGGER_LSTM_2_SIZE] = 256
    base_params[pkeys.FC_UNITS_1] = 128
    # Training parameters
    base_params[pkeys.EPOCHS_LR_UPDATE] = 5
    base_params[pkeys.MAX_LR_UPDATES] = 4

    # Data loading
    print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
    dataset = load_dataset(dataset_name, params=base_params)

    for cv_seed in cv_seed_list:
        for fold_id in fold_id_list:
            print('\nUsing CV seed %d and fold id %d' % (cv_seed, fold_id))
            # Generate split
            train_ids, val_ids, test_ids = dataset.cv_split(
                n_folds, fold_id, cv_seed, subject_ids=dataset.train_ids[1:])
            train_ids.append(dataset.train_ids[0])
            train_ids.sort()
            print("Subjects in each partition: train %d, val %d, test %d" % (
                len(train_ids), len(val_ids), len(test_ids)))
            # Compute global std
            fold_global_std = dataset.compute_global_std(np.concatenate([train_ids, val_ids]))
            dataset.global_std = fold_global_std
            print("Global STD set to %s" % fold_global_std)
            # Create data feeders
            data_train = FeederDataset(dataset, train_ids, task_mode, which_expert=which_expert)
            data_val = FeederDataset(dataset, val_ids, task_mode, which_expert=which_expert)
            data_test = FeederDataset(dataset, test_ids, task_mode, which_expert=which_expert)
            # Run CV
            for model_version, context_part in params_list:
                params = base_params.copy()
                params[pkeys.MODEL_VERSION] = model_version
                params[pkeys.BIGGER_CONTEXT_PART_OPTION] = context_part

                folder_name = '%s_cv%d_%s' % (
                    model_version,
                    cv_seed,
                    context_part
                )

                base_dir = os.path.join(
                    '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                    folder_name, 'fold%d' % fold_id)

                # Path to save results of run
                logdir = os.path.join(RESULTS_PATH, base_dir)
                print('This run directory: %s' % logdir)

                # Create and train model
                model = WaveletBLSTM(params=params, logdir=logdir)
                model.fit(data_train, data_val, verbose=verbose)
                # --------------  Predict
                # Save path for predictions
                save_dir = os.path.abspath(os.path.join(
                    RESULTS_PATH, 'predictions_%s' % dataset_name, base_dir))
                checks.ensure_directory(save_dir)
                feeders_dict = {
                    constants.TRAIN_SUBSET: data_train,
                    constants.VAL_SUBSET: data_val,
                    constants.TEST_SUBSET: data_test
                }
                for set_name in feeders_dict.keys():
                    print('Predicting %s' % set_name, flush=True)
                    data_inference = feeders_dict[set_name]
                    prediction = model.predict_dataset(data_inference, verbose=verbose)
                    filename = os.path.join(
                        save_dir,
                        'prediction_%s_%s.pkl' % (task_mode, set_name))
                    with open(filename, 'wb') as handle:
                        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Predictions saved at %s' % save_dir)
                print('')
