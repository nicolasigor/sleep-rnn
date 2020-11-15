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

    id_try_list = [1]

    # ----- Experiment settings
    experiment_name = 'deep_a7_grid_feats_windows'
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
    model_list = [
        constants.A7_V2
    ]
    window_duration_list = [0.3, 0.5, 1.0, 2.0]
    window_duration_abs_sig_pow_list = [0.15, 0.3, 0.5]

    params_list = list(itertools.product(
        model_list, window_duration_list, window_duration_abs_sig_pow_list))

    # Base parameters
    params = pkeys.default_params.copy()

    # Adjusted parameters (bsf)
    params[pkeys.A7_REMOVE_DELTA_IN_COV] = True
    params[pkeys.A7_USE_LOG_ABS_SIG_POW] = False
    params[pkeys.A7_USE_LOG_REL_SIG_POW] = True
    params[pkeys.A7_USE_LOG_SIG_COV] = False
    params[pkeys.A7_USE_LOG_SIG_CORR] = False

    # CNN parameters
    params[pkeys.A7_CNN_DROP_RATE] = 0.1
    params[pkeys.A7_CNN_N_LAYERS] = 6
    params[pkeys.A7_CNN_KERNEL_SIZE] = 7
    params[pkeys.A7_CNN_FILTERS] = 128

    # RNN parameters
    params[pkeys.A7_RNN_DROP_RATE] = 0.3
    params[pkeys.A7_RNN_LSTM_UNITS] = 128
    params[pkeys.A7_RNN_FC_UNITS] = 128

    # Other default values for A7 features
    params[pkeys.A7_USE_ZSCORE_REL_SIG_POW] = True
    params[pkeys.A7_USE_ZSCORE_SIG_COV] = True
    params[pkeys.A7_USE_ZSCORE_SIG_CORR] = False
    params[pkeys.A7_DISPERSION_MODE] = constants.DISPERSION_STD_ROBUST

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

                for model_version, window_duration, window_duration_abs_sig_pow in params_list:

                    params[pkeys.MODEL_VERSION] = model_version
                    params[pkeys.A7_WINDOW_DURATION] = window_duration
                    params[pkeys.A7_WINDOW_DURATION_ABS_SIG_POW] = window_duration_abs_sig_pow

                    folder_name = '%s_winDur%1.2f_winDurAbs%1.2f' % (
                        model_version, window_duration, window_duration_abs_sig_pow)

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
