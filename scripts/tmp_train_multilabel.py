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

    id_try_list = [0]

    # Grid parameters
    model_version_list = [
        constants.V11_MULTI
    ]

    # ----- Experiment settings
    experiment_name = "multilabel"
    task_mode = constants.N2_RECORD
    labels = ['ss_e1', 'ss_e2', 'kc_e1']
    label_data_dict = {
        'ss_e1': (constants.MASS_SS_NAME, 1),
        'ss_e2': (constants.MASS_SS_NAME, 2),
        'kc_e1': (constants.MASS_KC_NAME, 1),
    }
    description_str = 'Multilabel'
    verbose = True
    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.N_LABELS] = len(labels)
    params[pkeys.TYPE_LOSS] = constants.CROSS_ENTROPY_LOSS_MULTI

    for id_try in id_try_list:
        print('\nModel training on MASS with labels %s' % labels)
        all_data_dict = {'train': [], 'val': []}
        for label in labels:
            label_dataset_name = label_data_dict[label][0]
            label_which_expert = label_data_dict[label][1]
            print("\nLoading %s_e%d" % (label_dataset_name, label_which_expert))
            dataset = load_dataset(label_dataset_name, params=params)
            # Get training set ids
            all_train_ids = dataset.train_ids
            print('Using validation split %d' % id_try)
            # Generate split
            train_ids, val_ids = utils.split_ids_list_v2(
                all_train_ids, split_id=id_try)
            print('Training set IDs:', train_ids)
            data_train = FeederDataset(
                dataset, train_ids, task_mode, which_expert=label_which_expert)
            print('Validation set IDs:', val_ids)
            data_val = FeederDataset(
                dataset, val_ids, task_mode, which_expert=label_which_expert)
            all_data_dict['train'].append(data_train)
            all_data_dict['val'].append(data_val)

        for model_version in model_version_list:
            params[pkeys.MODEL_VERSION] = model_version
            folder_name = '%s' % model_version
            base_dir = os.path.join(
                '%s_%s' % (experiment_name, task_mode),
                folder_name, 'seed%d' % id_try)
            # Path to save results of run
            logdir = os.path.join(RESULTS_PATH, base_dir)
            print('\nThis run directory: %s' % logdir)

            # Create and train model
            model = WaveletBLSTM(params=params, logdir=logdir)
            model.fit(all_data_dict['train'], all_data_dict['val'], verbose=verbose)
    #
    #         # --------------  Predict
    #         # Save path for predictions
    #         save_dir = os.path.abspath(os.path.join(
    #             RESULTS_PATH, 'predictions_%s' % dataset_name,
    #             base_dir))
    #         checks.ensure_directory(save_dir)
    #
    #         feeders_dict = {
    #             constants.TRAIN_SUBSET: data_train,
    #             constants.TEST_SUBSET: data_test,
    #             constants.VAL_SUBSET: data_val
    #         }
    #         for set_name in feeders_dict.keys():
    #             print('Predicting %s' % set_name, flush=True)
    #             data_inference = feeders_dict[set_name]
    #             prediction = model.predict_dataset(
    #                 data_inference, verbose=verbose)
    #             filename = os.path.join(
    #                 save_dir,
    #                 'prediction_%s_%s.pkl' % (task_mode, set_name))
    #             with open(filename, 'wb') as handle:
    #                 pickle.dump(
    #                     prediction,
    #                     handle,
    #                     protocol=pickle.HIGHEST_PROTOCOL)
    #
    #             if set_name == constants.VAL_SUBSET:
    #                 # Validation AF1
    #                 # ----- Obtain AF1 metric
    #                 print('Computing Validation AF1...', flush=True)
    #                 detections_val = prediction.get_stamps()
    #                 events_val = data_val.get_stamps()
    #                 val_af1_at_half_thr = metrics.average_metric_with_list(
    #                     events_val, detections_val, verbose=False)
    #                 print('Validation AF1 with thr 0.5: %1.6f'
    #                       % val_af1_at_half_thr)
    #
    #                 metric_dict = {
    #                     'description': description_str,
    #                     'val_seed': id_try,
    #                     'database': dataset_name,
    #                     'task_mode': task_mode,
    #                     'val_af1': float(val_af1_at_half_thr)
    #                 }
    #                 with open(os.path.join(model.logdir, 'metric.json'),
    #                           'w') as outfile:
    #                     json.dump(metric_dict, outfile)
    #
    #         print('Predictions saved at %s' % save_dir)
    #         print('')
