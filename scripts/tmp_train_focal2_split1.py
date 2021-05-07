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
    fold_id_list = [i for i in range(n_folds)]
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1

    # ----- Experiment settings
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = 'focal2_%dfold-cv_exp%d' % (n_folds, which_expert)
    task_mode = constants.N2_RECORD
    description_str = 'experiments'
    verbose = True
    # Complement experiment folder name with date
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    power_base = 2
    model_version_list = [
        constants.V43
    ]
    cv_seed_list = [
        1
    ]
    focal_eps_list = [
        1.0, 0.2
    ]
    positive_weight_exponent_list = [
        0, -1, -2, -3
    ]
    params_list = list(itertools.product(
        model_version_list,
        focal_eps_list, positive_weight_exponent_list
    ))

    base_params = pkeys.default_params.copy()
    base_params[pkeys.BORDER_DURATION] = 3
    # BSF
    base_params[pkeys.SOFT_FOCAL_GAMMA] = 2.5

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
            for model_version, focal_eps, positive_weight_exponent in params_list:
                positive_weight = power_base ** positive_weight_exponent

                params = base_params.copy()
                params[pkeys.MODEL_VERSION] = model_version
                params[pkeys.SOFT_FOCAL_EPSILON] = focal_eps
                params[pkeys.CLASS_WEIGHTS] = [1.0, positive_weight]

                folder_name = '%s_cv%d_eps%1.1f_pos%+d' % (
                    model_version,
                    cv_seed,
                    focal_eps,
                    positive_weight_exponent
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
