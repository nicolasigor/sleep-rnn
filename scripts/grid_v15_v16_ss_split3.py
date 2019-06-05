from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import itertools
import json
import os
import pickle
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
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')
SEED_LIST = [123, 234, 345, 456]


if __name__ == '__main__':

    id_try_list = [3]
    # ----- Experiment settings
    experiment_name = 'grid_v15_V16'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]

    description_str = 'grid search for best mixed architecture'
    which_expert = 1
    verbose = True
    # -----

    version_cwtfilters_list = [
        (constants.V15, 32, 64),
        (constants.V15, 32, 32),
        (constants.V16, 64, 128)
    ]
    timefilters_list = [
        (64, 128, 256),
        (32, 64, 128)
    ]
    fb_init_list = [0.5, 1.0]
    parameter_list = itertools.product(
        version_cwtfilters_list, timefilters_list, fb_init_list
    )

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            print('\nModel training on %s_%s' % (dataset_name, task_mode))
            dataset = load_dataset(dataset_name)

            # Test set, used for predictions
            data_test = FeederDataset(
                dataset, dataset.test_ids, task_mode, which_expert=which_expert)

            # Get training set ids
            all_train_ids = dataset.train_ids
            for id_try in id_try_list:
                # Choose seed
                seed = SEED_LIST[id_try]
                print('\nUsing validation split seed %d' % seed)
                # Generate split
                train_ids, val_ids = utils.split_ids_list(
                    all_train_ids, seed=seed)
                print('Training set IDs:', train_ids)
                data_train = FeederDataset(
                    dataset, train_ids, task_mode, which_expert=which_expert)
                print('Validation set IDs:', val_ids)
                data_val = FeederDataset(
                    dataset, val_ids, task_mode, which_expert=which_expert)

                for version_cwtfilters, timefilters, fb_init in parameter_list:

                    version = version_cwtfilters[0]
                    cwt_filter_size_1 = version_cwtfilters[1]
                    cwt_filter_size_2 = version_cwtfilters[2]
                    time_filters_size_1 = timefilters[0]
                    time_filters_size_2 = timefilters[1]
                    time_filters_size_3 = timefilters[2]

                    folder_name = (
                            '%s_timef_%d_%d_%d_cwtf_%d_%d_fb_%s'
                            % (version,
                               time_filters_size_1,
                               time_filters_size_2,
                               time_filters_size_3,
                               cwt_filter_size_1,
                               cwt_filter_size_2,
                               fb_init))

                    # Path to save results of run
                    logdir = os.path.join(
                        RESULTS_PATH,
                        '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                        folder_name,
                        'seed%d' % id_try
                    )
                    print('This run directory: %s' % logdir)

                    # Create and train model
                    params = pkeys.default_params.copy()
                    params[pkeys.MODEL_VERSION] = version
                    params[pkeys.CWT_CONV_FILTERS_1] = cwt_filter_size_1
                    params[pkeys.CWT_CONV_FILTERS_2] = cwt_filter_size_2
                    params[pkeys.TIME_CONV_FILTERS_1] = time_filters_size_1
                    params[pkeys.TIME_CONV_FILTERS_2] = time_filters_size_2
                    params[pkeys.TIME_CONV_FILTERS_3] = time_filters_size_3
                    params[pkeys.FB_LIST] = [fb_init]

                    model = WaveletBLSTM(params, logdir=logdir)
                    model.fit(data_train, data_val, verbose=verbose)

                    # --------------  Predict
                    # Save path for predictions
                    save_dir = os.path.abspath(os.path.join(
                        RESULTS_PATH,
                        'predictions_%s' % dataset_name,
                        '%s_%s_train_%s'
                        % (experiment_name, task_mode, dataset_name),
                        folder_name,
                        'seed%d' % id_try
                    ))
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
                                'val_seed': seed,
                                'database': dataset_name,
                                'task_mode': task_mode,
                                'val_af1': float(val_af1_at_half_thr)
                            }
                            with open(os.path.join(model.logdir, 'metric.json'),
                                      'w') as outfile:
                                json.dump(metric_dict, outfile)

                    print('Predictions saved at %s' % save_dir)
                    print('')
