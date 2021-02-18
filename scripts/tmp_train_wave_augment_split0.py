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

    id_try_list = [0, 1, 2]

    train_fraction = 0.86

    # ----- Experiment settings
    experiment_name = 'wave_augment'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.CAP_ALL_SS_NAME
    ]
    which_expert = 1

    description_str = 'experiments'
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_version_list = [
        constants.V11_MKD2
    ]
    # params_list = list(itertools.product(model_version_list))

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
    # Training strategy
    params[pkeys.LEARNING_RATE] = 1e-4
    params[pkeys.MAX_ITERS] = 90000
    params[pkeys.ITERS_LR_UPDATE] = 3000
    # Wave Augment
    params[pkeys.AUG_RANDOM_WAVES_PROBA] = 1
    params[pkeys.AUG_RANDOM_ANTI_WAVES_PROBA] = 1

    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            print('\nModel training on %s_%s' % (dataset_name, task_mode))
            dataset = load_dataset(dataset_name, params=params)

            # Get training set ids
            all_train_ids = dataset.train_ids
            for id_try in id_try_list:
                print('\nUsing validation split %d' % id_try)
                # Generate split
                train_ids, val_ids = utils.split_ids_list_v2(
                    all_train_ids, split_id=id_try, train_fraction=train_fraction)

                print('Training set IDs:', train_ids)
                data_train = FeederDataset(
                    dataset, train_ids, task_mode, which_expert=which_expert)
                print('Validation set IDs:', val_ids)
                data_val = FeederDataset(
                    dataset, val_ids, task_mode, which_expert=which_expert)

                for model_version in model_version_list:

                    # Generate parameters for random wave augment
                    max_amplitude_list = [30, 20, 20, 20]  # uV, needs normalization
                    min_frequency_list = [1, 2, 5, 8]  # Hz
                    max_frequency_list = [2, 3, 7, 10]  # Hz
                    frequency_deviation_list = [0.5, 1, 2, 1]  # Hz
                    min_duration_list = [3, 2, 1, 1]  # s
                    max_duration_list = [5, 5, 5, 5]  # s
                    mask_list = [
                        constants.MASK_NONE,
                        constants.MASK_NONE,
                        constants.MASK_KEEP_BACKGROUND,
                        constants.MASK_KEEP_BACKGROUND,
                    ]
                    random_waves_params = [
                        dict(
                            max_amplitude=max_amplitude_list[i] / dataset.global_std,
                            min_frequency=min_frequency_list[i],
                            max_frequency=max_frequency_list[i],
                            frequency_deviation=frequency_deviation_list[i],
                            min_duration=min_duration_list[i],
                            max_duration=max_duration_list[i],
                            mask=mask_list[i])
                        for i in range(len(max_amplitude_list))
                    ]

                    # Generate parameters for random anti wave augment
                    max_attenuation_list = [0.5, 0.5, 1, 1, 1]  # [0, 1]
                    lowcut_list = [None, 2, 4, 7, 11]  # Hz
                    highcut_list = [2, 4, 7, 10, 16]  # Hz
                    min_duration_list = [3, 2, 1, 1, 1]  # s
                    max_duration_list = [5, 5, 5, 5, 5]  # s
                    mask_list = [
                        constants.MASK_NONE,
                        constants.MASK_NONE,
                        constants.MASK_KEEP_EVENTS,
                        constants.MASK_KEEP_EVENTS,
                        constants.MASK_KEEP_BACKGROUND]
                    random_anti_waves_params = [
                        dict(
                            max_attenuation=max_attenuation_list[i],
                            lowcut=lowcut_list[i],
                            highcut=highcut_list[i],
                            min_duration=min_duration_list[i],
                            max_duration=max_duration_list[i],
                            mask=mask_list[i])
                        for i in range(len(max_attenuation_list))
                    ]
                    params[pkeys.AUG_RANDOM_WAVES_PARAMS] = random_waves_params
                    params[pkeys.AUG_RANDOM_ANTI_WAVES_PARAMS] = random_anti_waves_params
                    params[pkeys.MODEL_VERSION] = model_version

                    folder_name = '%s' % (
                        model_version
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
