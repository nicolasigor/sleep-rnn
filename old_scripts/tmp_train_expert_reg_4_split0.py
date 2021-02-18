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


def generate_mkd_specs(multi_strategy_name, kernel_size, block_filters):
    if multi_strategy_name == 'dilated':
        mk_filters = [
            (kernel_size, block_filters // 2, 1),
            (kernel_size, block_filters // 4, 2),
            (kernel_size, block_filters // 8, 4),
            (kernel_size, block_filters // 8, 8),
        ]
    elif multi_strategy_name == 'none':
        mk_filters = [(kernel_size, block_filters, 1)]
    else:
        raise ValueError('strategy "%s" invalid' % multi_strategy_name)
    return mk_filters


if __name__ == '__main__':

    id_try_list = [0]

    # ----- Experiment settings
    experiment_name = 'expert_reg_4'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]
    which_expert = 1

    description_str = 'experiments'
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_version_list = [
        constants.V11_MKD2_EXPERTREG
    ]
    use_feats_list = [
        (True, True, False, False),
        (True, False, False, False),
        (False, True, False, False)
    ]
    base_power = 2
    coefficient_power_list = [
        -5, -6, -7, -8, -9
    ]
    window_duration_list = [
        1.0
    ]
    params_list = list(itertools.product(
        model_version_list, use_feats_list, coefficient_power_list, window_duration_list))

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.BORDER_DURATION] = 6

    # Segment net parameters
    params[pkeys.TIME_CONV_MK_PROJECT_FIRST] = False
    params[pkeys.TIME_CONV_MK_DROP_RATE] = 0.0
    params[pkeys.TIME_CONV_MK_SKIPS] = False
    params[pkeys.TIME_CONV_MKD_FILTERS_1] = generate_mkd_specs('none', 3, 64)
    params[pkeys.TIME_CONV_MKD_FILTERS_2] = generate_mkd_specs('dilated', 3, 128)
    params[pkeys.TIME_CONV_MKD_FILTERS_3] = generate_mkd_specs('dilated', 3, 256)
    params[pkeys.FC_UNITS] = 128

    # Expert branch parameters
    params[pkeys.EXPERT_BRANCH_REL_POWER_BROAD_LOWCUT] = 4  # In paper is 4.5Hz
    params[pkeys.EXPERT_BRANCH_COVARIANCE_BROAD_LOWCUT] = 4  # In paper is None
    params[pkeys.EXPERT_BRANCH_ZSCORE_DISPERSION_MODE] = constants.DISPERSION_STD_ROBUST
    params[pkeys.EXPERT_BRANCH_REL_POWER_USE_ZSCORE] = True
    params[pkeys.EXPERT_BRANCH_COVARIANCE_USE_ZSCORE] = True
    params[pkeys.EXPERT_BRANCH_CORRELATION_USE_ZSCORE] = False
    params[pkeys.EXPERT_BRANCH_COLLAPSE_TIME_MODE] = None
    params[pkeys.EXPERT_BRANCH_CORRELATION_TRANSFORMATION] = None
    params[pkeys.EXPERT_BRANCH_ABS_POWER_TRANSFORMATION] = 'log'
    params[pkeys.EXPERT_BRANCH_REL_POWER_TRANSFORMATION] = 'log'
    params[pkeys.EXPERT_BRANCH_COVARIANCE_TRANSFORMATION] = 'log'

    params[pkeys.EXPERT_BRANCH_REGRESSION_FROM_BLSTM] = True
    params[pkeys.EXPERT_BRANCH_REGRESSION_HIDDEN_UNITS] = 128

    params[pkeys.TYPE_LOSS] = constants.CROSS_ENTROPY_LOSS_WITH_SELF_SUPERVISION

    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            print('\nModel training on %s_%s' % (dataset_name, task_mode))
            dataset = load_dataset(dataset_name, params=params)

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

                for model_version, use_feats, coefficient_power, window_duration in params_list:

                    params[pkeys.MODEL_VERSION] = model_version

                    params[pkeys.EXPERT_BRANCH_USE_ABS_POWER] = use_feats[0]
                    params[pkeys.EXPERT_BRANCH_USE_REL_POWER] = use_feats[1]
                    params[pkeys.EXPERT_BRANCH_USE_COVARIANCE] = use_feats[2]
                    params[pkeys.EXPERT_BRANCH_USE_CORRELATION] = use_feats[3]
                    params[pkeys.EXPERT_BRANCH_REGRESSION_LOSS_COEFFICIENT] = base_power ** coefficient_power
                    params[pkeys.EXPERT_BRANCH_WINDOW_DURATION] = window_duration  # For expert reg it is fixed.

                    folder_name = '%s_feats%d%d%d%d_pow(%d,%d)_w%1.1fs' % (
                        model_version, use_feats[0], use_feats[1], use_feats[2], use_feats[3],
                        base_power, coefficient_power, window_duration
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