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


def mkfilters_to_str(list_of_tuples):
    name = '('
    for k, f in list_of_tuples:
        name = name + ('%d:%d,' % (k, f))
    name = name + ')'
    return name


def generate_mk_specs(multi_strategy_name, block_filters):
    if multi_strategy_name == 'linear':
        mk_filters = [
            (3, block_filters // 2),
            (5, block_filters // 4),
            (7, block_filters // 8),
            (9, block_filters // 8),
        ]
    elif multi_strategy_name == 'exp1':
        mk_filters = [
            (3, block_filters // 2),
            (5, block_filters // 4),
            (9, block_filters // 8),
            (17, block_filters // 8),
        ]
    elif multi_strategy_name == 'exp2':
        mk_filters = [
            (3, block_filters // 2),
            (7, block_filters // 4),
            (15, block_filters // 8),
            (31, block_filters // 8),
        ]
    elif multi_strategy_name == 'none':
        mk_filters = [(3, block_filters)]
    else:
        raise ValueError('strategy "%s" invalid' % multi_strategy_name)
    return mk_filters


if __name__ == '__main__':

    id_try_list = [0, 1]

    # ----- Experiment settings
    experiment_name = 'mkconv'
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
    model_list = [constants.V11_MK]
    init_filters_list = [64]
    first_block_behavior_list = [
        'multi',
        # 'single'
    ]
    multi_strategy_list = [
        # 'exp2',
        'exp1',
        # 'linear',
        # 'none'
    ]
    project_first_list = [
        False,
        True
    ]
    drop_conv_rate_list = [
        0.0,
        0.1
    ]
    conv_skip_list = [
        True,
        False
    ]
    params_list = list(itertools.product(
        model_list,
        conv_skip_list, drop_conv_rate_list, project_first_list,
        init_filters_list, first_block_behavior_list, multi_strategy_list
    ))

    # Base parameters
    params = pkeys.default_params.copy()

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

                for model_version, conv_skip, drop_conv_rate, project_first, init_filters, first_block_behavior, multi_strategy  in params_list:
                    if first_block_behavior == 'single':
                        mk_filters_1 = generate_mk_specs('none', init_filters)
                    elif multi_strategy != 'none':
                        mk_filters_1 = generate_mk_specs(multi_strategy, init_filters)
                    else:
                        continue
                    mk_filters_2 = generate_mk_specs(multi_strategy, 2 * init_filters)
                    mk_filters_3 = generate_mk_specs(multi_strategy, 4 * init_filters)

                    params[pkeys.MODEL_VERSION] = model_version
                    params[pkeys.TIME_CONV_MK_FILTERS_1] = mk_filters_1
                    params[pkeys.TIME_CONV_MK_FILTERS_2] = mk_filters_2
                    params[pkeys.TIME_CONV_MK_FILTERS_3] = mk_filters_3
                    params[pkeys.TIME_CONV_MK_PROJECT_FIRST] = project_first
                    params[pkeys.TIME_CONV_MK_DROP_RATE] = drop_conv_rate
                    params[pkeys.TIME_CONV_MK_SKIPS] = conv_skip

                    folder_name = '%s_skip%d_dr%1.1f_p%d_f%d_multi-%s_first-%s' % (
                        model_version,
                        conv_skip, drop_conv_rate, project_first,
                        init_filters, multi_strategy, first_block_behavior)

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
