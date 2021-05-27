from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
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
    experiment_name = 'thesis_ablation_v3_%dfold-cv_exp%d' % (n_folds, which_expert)
    task_mode = constants.N2_RECORD
    description_str = 'experiments'
    verbose = True
    # Complement experiment folder name with date
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_version_list = [
        constants.V2_TIME
    ]
    cv_seed_list = [
        0, 1, 2
    ]
    loss_augment_list = [
        # ('xent', 0),
        # ('xent', 1),
        # ('soft', 0),
        ('soft', 1),
    ]
    params_list = list(itertools.product(
        model_version_list, loss_augment_list
    ))

    base_params = pkeys.default_params.copy()
    base_params[pkeys.BORDER_DURATION] = pkeys.DEFAULT_BORDER_DURATION_V2_TIME
    # Default parameters with magnitudes in microvolts (uv)
    da_unif_noise_intens_uv = pkeys.DEFAULT_AUG_INDEP_UNIFORM_NOISE_INTENSITY_MICROVOLTS
    da_random_waves_map = {
        constants.SPINDLE: pkeys.DEFAULT_AUG_RANDOM_WAVES_PARAMS_SPINDLE,
        constants.KCOMPLEX: pkeys.DEFAULT_AUG_RANDOM_WAVES_PARAMS_KCOMPLEX}
    da_random_antiwaves_map = {
        constants.SPINDLE: pkeys.DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_SPINDLE,
        constants.KCOMPLEX: pkeys.DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_KCOMPLEX}

    # Data loading
    print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
    dataset = load_dataset(dataset_name, params=base_params)

    for cv_seed in cv_seed_list:
        for fold_id in fold_id_list:
            fold_id_to_save = cv_seed * n_folds + fold_id
            print('\nUsing CV seed %d and fold id %d (%d)' % (cv_seed, fold_id, fold_id_to_save))
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
            # First prepare parameters that depend on global std
            base_params[pkeys.AUG_INDEP_UNIFORM_NOISE_INTENSITY] = da_unif_noise_intens_uv / dataset.global_std
            da_random_waves = copy.deepcopy(da_random_waves_map[dataset.event_name])
            da_random_antiwaves = copy.deepcopy(da_random_antiwaves_map[dataset.event_name])
            for da_id in range(len(da_random_waves)):
                da_random_waves[da_id]['max_amplitude'] = da_random_waves[da_id]['max_amplitude_microvolts'] / dataset.global_std
                da_random_waves[da_id].pop('max_amplitude_microvolts')
            base_params[pkeys.AUG_RANDOM_WAVES_PARAMS] = da_random_waves
            base_params[pkeys.AUG_RANDOM_ANTI_WAVES_PARAMS] = da_random_antiwaves
            if dataset_name == constants.INTA_SS_NAME:
                base_params.update(pkeys.DEFAULT_INTA_POSTPROCESSING_PARAMS)
            # Run CV
            for model_version, loss_augment in params_list:
                # Now set parameters for this run
                loss_name = loss_augment[0]
                wave_augment_proba = loss_augment[1]
                params = copy.deepcopy(base_params)
                params[pkeys.MODEL_VERSION] = model_version
                params[pkeys.AUG_RANDOM_WAVES_PROBA] = wave_augment_proba
                params[pkeys.AUG_RANDOM_ANTI_WAVES_PROBA] = wave_augment_proba
                if loss_name == 'xent':
                    params[pkeys.SOFT_FOCAL_EPSILON] = 1.0
                    params[pkeys.CLASS_WEIGHTS] = [1.0, 1.0]

                folder_name = '%s_loss-%s_wave%d' % (
                    model_version,
                    loss_name,
                    wave_augment_proba
                )

                base_dir = os.path.join(
                    '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                    folder_name, 'fold%d' % fold_id_to_save)

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
