from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models_mod import WaveletBLSTMMod
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')
SEED_LIST = [123, 234, 345, 456]


if __name__ == '__main__':

    id_try_list = [0, 1, 2, 3]

    # ----- Experiment settings
    experiment_name = 'bsf_cwt_norm'
    task_mode = constants.WN_RECORD

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]

    description_str = 'bsf cwt modification, full bsf'
    which_expert = 1
    verbose = True

    max_iters = 30000  # For debugging reasons
    # -----

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    for dataset_name in dataset_name_list:
        print('\nModel training on %s_%s' % (dataset_name, task_mode))
        dataset = load_dataset(dataset_name)
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

            # Path to save results of run
            logdir = os.path.join(
                RESULTS_PATH,
                '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                'bsf',
                'seed%d' % id_try
            )
            print('This run directory: %s' % logdir)

            # Create and train model
            params = pkeys.default_params.copy()

            params[pkeys.MAX_ITERS] = max_iters
            params[pkeys.TRAINABLE_WAVELET] = False

            model = WaveletBLSTMMod(params, logdir=logdir)
            model.fit(data_train, data_val, verbose=verbose)

            # Validation metrics
            print('Predicting Validation set')
            prediction_val = model.predict_dataset(
                data_val, verbose=verbose)
            print('Done set')

            # ----- Obtain AF1 metric
            print('Computing AF1...', flush=True)
            detections_val = prediction_val.get_stamps()
            events_val = data_val.get_stamps()
            val_af1_at_half_thr = metrics.average_metric_with_list(
                events_val, detections_val, verbose=False)
            print('Validation AF1 with thr 0.5: %1.6f' % val_af1_at_half_thr)

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

            print('')
