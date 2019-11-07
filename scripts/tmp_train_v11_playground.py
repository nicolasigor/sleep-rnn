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
from sleeprnn.detection.threshold_optimization import get_optimal_threshold

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try_list = [0]

    # ----- Experiment settings
    experiment_name = 'v11_playground'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]

    description_str = 'playground'

    folder_name = 'elastic_test'

    which_expert = 1
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.MODEL_VERSION] = constants.V11
    # params[pkeys.TOTAL_DOWNSAMPLING_FACTOR] = 8
    # params[pkeys.LAST_OUTPUT_CONV_FILTERS] = 32
    params[pkeys.ITERS_LR_UPDATE] = 500
    params[pkeys.MAX_ITERS] = 5000
    params[pkeys.FS] = 200
    params[pkeys.LR_UPDATE_FACTOR] = 0.5
    params[pkeys.MAX_LR_UPDATES] = 3
    # params[pkeys.LR_UPDATE_CRITERION] = constants.LOSS_CRITERION
    params[pkeys.LR_UPDATE_RESET_OPTIMIZER] = True
    params[pkeys.KEEP_BEST_VALIDATION] = True
    params[pkeys.AUG_GAUSSIAN_NOISE_PROBA] = 0.5
    params[pkeys.AUG_ELASTIC_PROBA] = 0.5
    params[pkeys.AUG_ELASTIC_ALPHA] = 0.2
    params[pkeys.AUG_ELASTIC_SIGMA] = 0.05

    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            print('\nModel training on %s_%s' % (dataset_name, task_mode))
            if params[pkeys.FS] == 200:
                load_checkpoint = True
            else:
                load_checkpoint = False
            dataset = load_dataset(dataset_name, load_checkpoint=load_checkpoint, params=params)

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

                predicted_dict = {}
                for set_name in feeders_dict.keys():
                    print('Predicting %s' % set_name, flush=True)
                    data_inference = feeders_dict[set_name]
                    prediction = model.predict_dataset(
                        data_inference, verbose=verbose)
                    predicted_dict[set_name] = prediction

                # Find best thr
                feeder_dataset_list = []
                predicted_dataset_list = []
                for set_name in [constants.TRAIN_SUBSET, constants.VAL_SUBSET]:
                    data_inference = feeders_dict[set_name]
                    feeder_dataset_list.append(data_inference)
                    prediction_obj = predicted_dict[set_name]
                    predicted_dataset_list.append(prediction_obj)
                best_thr = get_optimal_threshold(
                    feeder_dataset_list,
                    predicted_dataset_list)
                print('Best thr %1.2f' % best_thr)

                for set_name in feeders_dict.keys():
                    prediction = predicted_dict[set_name]
                    prediction.set_probability_threshold(best_thr)

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
                        print('Computing %s AF1...' % set_name, flush=True)
                        detections_val = prediction.get_stamps()
                        events_val = feeders_dict[set_name].get_stamps()
                        val_af1_at_half_thr = metrics.average_metric_with_list(
                            events_val, detections_val, verbose=False)
                        print('Validation AF1 with best thr: %1.6f'
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

                    if set_name == constants.TEST_SUBSET:
                        print('Computing %s AF1...' % set_name, flush=True)
                        detections_val = prediction.get_stamps()
                        events_val = feeders_dict[set_name].get_stamps()
                        val_af1_at_half_thr = metrics.average_metric_with_list(
                            events_val, detections_val, verbose=False)
                        print('Test AF1 with best thr: %1.6f'
                              % val_af1_at_half_thr)

                print('Predictions saved at %s' % save_dir)
                print('')
