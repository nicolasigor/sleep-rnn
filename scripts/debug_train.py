from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
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

    # ----- Experiment settings
    experiment_name = 'debug'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]

    description_str = 'bsf'
    which_expert = 1
    verbose = True
    # -----

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
            # Choose seed
            seed = SEED_LIST[0]
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
            )
            print('This run directory: %s' % logdir)

            # Create and train model
            params = pkeys.default_params.copy()

            params[pkeys.MODEL_VERSION] = constants.DEBUG
            params[pkeys.MAX_ITERS] = 10000

            model = WaveletBLSTM(params, logdir=logdir)
            model.fit(data_train, data_val, verbose=verbose)
