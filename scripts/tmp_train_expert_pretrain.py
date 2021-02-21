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

    # ----- Experiment settings
    experiment_name = 'expert_pretrain'
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
    type_loss_list = [
        'xentropy',
        'focal'
    ]
    params_list = list(itertools.product(model_version_list, type_loss_list))

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
    params[pkeys.PRETRAIN_MAX_LR_UPDATES] = 4
    params[pkeys.PRETRAIN_ITERS_INIT] = 17500
    params[pkeys.PRETRAIN_ITERS_ANNEAL] = 3500

    # Soft focal loss setting
    params[pkeys.ANTIBORDER_AMPLITUDE] = 0
    params[pkeys.ANTIBORDER_HALF_WIDTH] = 6
    params[pkeys.SOFT_FOCAL_GAMMA] = 3.0
    params[pkeys.SOFT_FOCAL_EPSILON] = 0.5
    params[pkeys.CLASS_WEIGHTS] = [1.0, 1.0]

    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            print('\nModel training on %s_%s' % (dataset_name, task_mode))
            dataset = load_dataset(dataset_name, params=params)
            all_train_ids = dataset.train_ids
            print('Training set IDs:', all_train_ids)
            data_train = FeederDataset(dataset, all_train_ids, task_mode, which_expert=which_expert)

            for model_version, type_loss in params_list:
                params[pkeys.MODEL_VERSION] = model_version
                if type_loss == 'xentropy':
                    params[pkeys.TYPE_LOSS] = constants.CROSS_ENTROPY_LOSS
                elif type_loss == 'focal':
                    params[pkeys.TYPE_LOSS] = constants.WEIGHTED_CROSS_ENTROPY_LOSS_V5
                else:
                    raise ValueError()

                folder_name = '%s_%s' % (model_version, type_loss)

                base_dir = os.path.join(
                    '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                    folder_name)

                # Path to save results of run
                logdir = os.path.join(RESULTS_PATH, base_dir)
                print('This run directory: %s' % logdir)

                # Create and train model
                model = WaveletBLSTM(params=params, logdir=logdir)
                model.fit_without_validation(data_train, verbose=verbose)
