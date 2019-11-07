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

    id_try_list = [3]

    # ----- Experiment settings
    experiment_name = 'v22_grid'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME,
        constants.MASS_KC_NAME
    ]

    description_str = 'v22 (cwt indep)'
    which_expert = 1
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    return_real_imag_magn_phase_list = [
        (True, True, True, False),
        (True, True, False, False)
    ]
    drop_rate_before_lstm_list = [0.0, 0.3]
    cwt_filters_3_list = [64, 32]

    parameters_list = list(itertools.product(
        return_real_imag_magn_phase_list,
        drop_rate_before_lstm_list,
        cwt_filters_3_list))

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.NORM_COMPUTATION_MODE] = constants.NORM_GLOBAL
    params[pkeys.MODEL_VERSION] = constants.V22
    params[pkeys.CWT_CONV_FILTERS_1] = 32
    params[pkeys.CWT_CONV_FILTERS_2] = 32
    params[pkeys.USE_RELU] = None
    params[pkeys.USE_LOG] = None
    params[pkeys.FB_LIST] = [0.5]

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

                for return_real_imag_magn_phase, drop_rate_before_lstm, cwt_filters_3 in parameters_list:

                    return_real_part = return_real_imag_magn_phase[0]
                    return_imag_part = return_real_imag_magn_phase[1]
                    return_magnitude = return_real_imag_magn_phase[2]
                    return_phase = return_real_imag_magn_phase[3]

                    params[pkeys.CWT_RETURN_REAL_PART] = return_real_part
                    params[pkeys.CWT_RETURN_IMAG_PART] = return_imag_part
                    params[pkeys.CWT_RETURN_MAGNITUDE] = return_magnitude
                    params[pkeys.CWT_RETURN_PHASE] = return_phase
                    params[pkeys.DROP_RATE_BEFORE_LSTM] = drop_rate_before_lstm
                    params[pkeys.CWT_CONV_FILTERS_3] = cwt_filters_3

                    folder_name = 'r_%d_i_%d_m_%d_p_%d_drop_%s_f_%d' % (
                        int(return_real_part),
                        int(return_imag_part),
                        int(return_magnitude),
                        int(return_phase),
                        drop_rate_before_lstm,
                        cwt_filters_3)

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
