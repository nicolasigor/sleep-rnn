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
    folds = 4
    dataset_name = constants.MASS_SS_ALT_NAME
    which_expert_list = [1]

    id_try_list = [0]  # [i for i in range(folds)]
    train_fraction = (folds - 1) / folds

    this_date = datetime.datetime.now().strftime("%Y%m%d")
    for which_expert in which_expert_list:

        # ----- Experiment settings
        experiment_name = 'check_contralateral_exp%d' % which_expert
        task_mode = constants.N2_RECORD
        description_str = 'experiments'
        verbose = True

        # Complement experiment folder name with date
        experiment_name = '%s_%s' % (this_date, experiment_name)

        # Grid parameters
        model_version_list = [
            constants.V11_MKD2
        ]

        # Base parameters
        params = pkeys.default_params.copy()
        params[pkeys.BORDER_DURATION] = 4
        params[pkeys.TIME_CONV_MK_PROJECT_FIRST] = False
        params[pkeys.TIME_CONV_MK_DROP_RATE] = 0.0
        params[pkeys.TIME_CONV_MK_SKIPS] = False
        params[pkeys.TIME_CONV_MKD_FILTERS_1] = generate_mkd_specs('none', 3, 64)
        params[pkeys.TIME_CONV_MKD_FILTERS_2] = generate_mkd_specs('dilated', 3, 128)
        params[pkeys.TIME_CONV_MKD_FILTERS_3] = generate_mkd_specs('dilated', 3, 256)

        # Training settings
        params[pkeys.MAX_EPOCHS] = 100
        params[pkeys.EPOCHS_LR_UPDATE] = 5
        params[pkeys.MAX_LR_UPDATES] = 4

        print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
        dataset = load_dataset(dataset_name, params=params)

        # Test set, used for predictions
        data_test = FeederDataset(dataset, dataset.test_ids, task_mode, which_expert=which_expert)

        for id_try in id_try_list:
            print('\nUsing validation split %d' % id_try)
            # Generate split
            train_ids, val_ids = utils.split_ids_list_v2(
                dataset.train_ids, split_id=id_try, train_fraction=train_fraction)
            print('Training set IDs:', train_ids)
            data_train = FeederDataset(
                dataset, train_ids, task_mode, which_expert=which_expert)
            print('Validation set IDs:', val_ids)
            data_val = FeederDataset(
                dataset, val_ids, task_mode, which_expert=which_expert)

            for model_version in model_version_list:

                params[pkeys.MODEL_VERSION] = model_version

                folder_name = '%s' % model_version

                base_dir = os.path.join(
                    '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
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
