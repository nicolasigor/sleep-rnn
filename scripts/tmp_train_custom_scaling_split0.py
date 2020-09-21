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

from sleeprnn.data import utils
from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try_list = [0, 1]

    # ----- Experiment settings
    experiment_name = 'custom_scaling'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME,
    ]
    which_expert = 1

    description_str = 'experiments'
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_list = [
        constants.V19
    ]
    prop_factor_list = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.NORM_COMPUTATION_MODE] = constants.NORM_GLOBAL_CUSTOM

    ref_precision_dict = {
        1: 0.9295302013422819,
        3: 0.7533333333333333,
        5: 0.8846153846153846,
        7: 0.7693920335429769,
        9: 0.9164305949008499,
        10: 0.8053097345132744,
        11: 0.7038895859473023,
        14: 0.6242038216560509,
        17: 0.8162839248434238,
        18: 0.877720207253886,
        19: 0.6870588235294117,
        2: 0,
        6: 0,
        12: 0,
        13: 0
    }

    ref_recall_dict = {
        1: 0.7959770114942529,
        3: 0.8014184397163121,
        5: 0.6197604790419161,
        7: 0.8137472283813747,
        9: 0.8017348203221809,
        10: 0.9180327868852459,
        11: 0.9288079470198676,
        14: 0.9730496453900709,
        17: 0.8354700854700855,
        18: 0.7378048780487805,
        19: 0.9299363057324841,
        2: 0,
        6: 0,
        12: 0,
        13: 0
    }
    scaling_test_ids = [2, 6, 12, 13]

    for task_mode in task_mode_list:
        for prop_factor in prop_factor_list:

            # Build custom scaling dictionary
            custom_scaling_dict = {}
            mean_scaling_factor = 0
            for subject_id in ref_precision_dict.keys():
                precision = ref_precision_dict[subject_id]
                recall = ref_recall_dict[subject_id]
                difference = recall - precision
                custom_scaling_dict[subject_id] = 1.0 - prop_factor * difference
                if subject_id not in scaling_test_ids:
                    mean_scaling_factor += custom_scaling_dict[subject_id]
            mean_scaling_factor /= 11
            # Normalize custom scaling dictionary
            for subject_id in custom_scaling_dict.keys():
                if subject_id not in scaling_test_ids:
                    custom_scaling_dict[subject_id] /= mean_scaling_factor

            for dataset_name in dataset_name_list:
                print('\nModel training on %s_%s' % (dataset_name, task_mode))
                dataset = load_dataset(dataset_name, params=params, custom_scaling_dict=custom_scaling_dict)

                # Test set, used for predictions
                data_test = FeederDataset(
                    dataset, dataset.test_ids, task_mode, which_expert=which_expert)

                # Get training set ids
                all_train_ids = dataset.train_ids
                for id_try in id_try_list:
                    print('\nUsing validation split %d' % id_try)
                    # Generate split
                    train_ids, val_ids = utils.split_ids_list_v2(
                        all_train_ids, split_id=id_try)

                    print('Training set IDs:', train_ids)
                    data_train = FeederDataset(
                        dataset, train_ids, task_mode, which_expert=which_expert)
                    print('Validation set IDs:', val_ids)
                    data_val = FeederDataset(
                        dataset, val_ids, task_mode, which_expert=which_expert)

                    for model_version in model_list:

                        params[pkeys.MODEL_VERSION] = model_version

                        folder_name = '%s_prop%1.2f' % (
                            model_version, prop_factor
                        )

                        base_dir = os.path.join(
                            '%s_%s_train_%s' % (
                                experiment_name, task_mode, dataset_name),
                            folder_name, 'seed%d' % id_try)

                        # Path to save results of run
                        logdir = os.path.join(RESULTS_PATH, base_dir)
                        print('This run directory: %s' % logdir)

                        # Create and train model
                        model = WaveletBLSTM(params=params, logdir=logdir)
                        model.fit(data_train, data_val, verbose=verbose)

                        # --------------  Predict
                        # Save path for predictions
                        save_dir = os.path.abspath(os.path.join(
                            RESULTS_PATH, 'predictions_%s' % dataset_name,
                            base_dir))
                        checks.ensure_directory(save_dir)

                        feeders_dict = {
                            constants.TRAIN_SUBSET: data_train,
                            constants.TEST_SUBSET: data_test,
                            constants.VAL_SUBSET: data_val
                        }
                        for set_name in feeders_dict.keys():
                            print('Predicting %s' % set_name, flush=True)
                            data_inference = feeders_dict[set_name]
                            prediction = model.predict_dataset(
                                data_inference, verbose=verbose)
                            filename = os.path.join(
                                save_dir,
                                'prediction_%s_%s.pkl' % (task_mode, set_name))
                            with open(filename, 'wb') as handle:
                                pickle.dump(
                                    prediction,
                                    handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)

                            if set_name == constants.VAL_SUBSET:
                                # Validation AF1
                                # ----- Obtain AF1 metric
                                print('Computing Validation AF1...', flush=True)
                                detections_val = prediction.get_stamps()
                                events_val = data_val.get_stamps()
                                val_af1_at_half_thr = metrics.average_metric_with_list(
                                    events_val, detections_val, verbose=False)
                                print('Validation AF1 with thr 0.5: %1.6f'
                                      % val_af1_at_half_thr)

                                metric_dict = {
                                    'description': description_str,
                                    'val_seed': id_try,
                                    'database': dataset_name,
                                    'task_mode': task_mode,
                                    'val_af1': float(val_af1_at_half_thr)
                                }
                                with open(os.path.join(model.logdir, 'metric.json'),
                                          'w') as outfile:
                                    json.dump(metric_dict, outfile)

                        print('Predictions saved at %s' % save_dir)
                        print('')
