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

from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    this_date = datetime.datetime.now().strftime("%Y%m%d")

    for which_expert in [2]:

        # ----- Experiment settings
        experiment_name = 'lego_2_pretrain_v2_exp%d' % which_expert
        task_mode = constants.N2_RECORD
        dataset_name = constants.CAP_SS_NAME
        description_str = 'experiments'
        verbose = True

        # Complement experiment folder name with date
        experiment_name = '%s_%s' % (this_date, experiment_name)

        # Grid parameters
        model_version_list = [
            constants.V43
        ]

        conv_part_list = [
            # ('multi-d8', {
            #     pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'multi_dilated',
            #     pkeys.BIGGER_STEM_KERNEL_SIZE: 3,
            #     pkeys.BIGGER_STEM_FILTERS: 64,
            #     pkeys.BIGGER_MAX_DILATION: 8,
            # }),
            # ('res-multi-d4', {
            #     pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual_multi_dilated',
            #     pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            #     pkeys.BIGGER_STEM_FILTERS: 32,
            #     pkeys.BIGGER_MAX_DILATION: 4,
            #     pkeys.BIGGER_STAGE_1_SIZE: 1,
            #     pkeys.BIGGER_STAGE_2_SIZE: 1,
            #     pkeys.BIGGER_STAGE_3_SIZE: 1,
            # }),
            ('res-multi-d8', {
                pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual_multi_dilated',
                pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
                pkeys.BIGGER_STEM_FILTERS: 64,
                pkeys.BIGGER_MAX_DILATION: 8,
                pkeys.BIGGER_STAGE_1_SIZE: 1,
                pkeys.BIGGER_STAGE_2_SIZE: 1,
                pkeys.BIGGER_STAGE_3_SIZE: 0,
            }),
            # ('res-d1', {
            #     pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual',
            #     pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            #     pkeys.BIGGER_STEM_FILTERS: 32,
            #     pkeys.BIGGER_MAX_DILATION: 1,
            #     pkeys.BIGGER_STAGE_1_SIZE: 2,
            #     pkeys.BIGGER_STAGE_2_SIZE: 2,
            #     pkeys.BIGGER_STAGE_3_SIZE: 4,
            # }),
            # ('res-d4', {
            #     pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual',
            #     pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            #     pkeys.BIGGER_STEM_FILTERS: 32,
            #     pkeys.BIGGER_MAX_DILATION: 4,
            #     pkeys.BIGGER_STAGE_1_SIZE: 2,
            #     pkeys.BIGGER_STAGE_2_SIZE: 2,
            #     pkeys.BIGGER_STAGE_3_SIZE: 3,
            # }),
            # ('res-d8', {
            #     pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual',
            #     pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            #     pkeys.BIGGER_STEM_FILTERS: 64,
            #     pkeys.BIGGER_MAX_DILATION: 8,
            #     pkeys.BIGGER_STAGE_1_SIZE: 2,
            #     pkeys.BIGGER_STAGE_2_SIZE: 4,
            #     pkeys.BIGGER_STAGE_3_SIZE: 0,
            # }),
        ]

        context_part_list = [
            # ('att2', {
            #     pkeys.BIGGER_CONTEXT_PART_OPTION: 'attention',
            #     pkeys.BIGGER_ATT_N_BLOCKS: 2,
            #     pkeys.BIGGER_ATT_TYPE_NORM: 'layernorm'
            # }),
            # ('att3', {
            #     pkeys.BIGGER_CONTEXT_PART_OPTION: 'attention',
            #     pkeys.BIGGER_ATT_N_BLOCKS: 3,
            #     pkeys.BIGGER_ATT_TYPE_NORM: 'layernorm'
            # }),
            ('res-lstm', {
                pkeys.BIGGER_CONTEXT_PART_OPTION: 'residual_lstm',
                pkeys.BIGGER_LSTM_1_SIZE: 256,
                pkeys.BIGGER_LSTM_2_SIZE: 256,
                pkeys.FC_UNITS: 128,
                pkeys.BIGGER_ATT_TYPE_NORM: None
            }),
            ('lstm', {
                pkeys.BIGGER_CONTEXT_PART_OPTION: 'lstm',
                pkeys.BIGGER_LSTM_1_SIZE: 256,
                pkeys.BIGGER_LSTM_2_SIZE: 256,
                pkeys.FC_UNITS: 128
            }),
            ('res-lstm-ln', {
                pkeys.BIGGER_CONTEXT_PART_OPTION: 'residual_lstm',
                pkeys.BIGGER_LSTM_1_SIZE: 256,
                pkeys.BIGGER_LSTM_2_SIZE: 256,
                pkeys.FC_UNITS: 128,
                pkeys.BIGGER_ATT_TYPE_NORM: 'layernorm'
            }),
            # ('res-lstm-bn', {
            #     pkeys.BIGGER_CONTEXT_PART_OPTION: 'residual_lstm',
            #     pkeys.BIGGER_LSTM_1_SIZE: 256,
            #     pkeys.BIGGER_LSTM_2_SIZE: 256,
            #     pkeys.FC_UNITS: 128,
            #     pkeys.BIGGER_ATT_TYPE_NORM: 'batchnorm'
            # }),
        ]

        params_list = list(itertools.product(
            model_version_list, conv_part_list, context_part_list
        ))

        # Common parameters
        base_params = pkeys.default_params.copy()
        base_params[pkeys.BORDER_DURATION] = 4
        base_params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE] = 3
        base_params[pkeys.ATT_DROP_RATE] = 0.1
        base_params[pkeys.ATT_DIM] = 256
        base_params[pkeys.ATT_N_HEADS] = 4
        base_params[pkeys.ATT_PE_FACTOR] = 10000
        base_params[pkeys.DROP_RATE_BEFORE_LSTM] = 0.2
        base_params[pkeys.DROP_RATE_HIDDEN] = 0.2

        base_params[pkeys.MAX_EPOCHS] = 100
        base_params[pkeys.EPOCHS_LR_UPDATE] = 4
        base_params[pkeys.MAX_LR_UPDATES] = 3

        # Training strategy
        base_params[pkeys.PRETRAIN_EPOCHS_INIT] = 20
        base_params[pkeys.PRETRAIN_EPOCHS_ANNEAL] = 4
        base_params[pkeys.PRETRAIN_MAX_LR_UPDATES] = 3

        print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
        dataset = load_dataset(dataset_name, params=base_params)
        all_train_ids = dataset.train_ids
        print('Training set IDs:', all_train_ids)
        data_train = FeederDataset(dataset, all_train_ids, task_mode, which_expert=which_expert)

        for model_version, conv_part, context_part in params_list:
            conv_part_name, conv_part_params = conv_part
            context_part_name, context_part_params = context_part
            params = base_params.copy()
            params.update(conv_part_params)
            params.update(context_part_params)
            params[pkeys.MODEL_VERSION] = model_version
            folder_name = '%s_%s_%s' % (
                model_version,
                conv_part_name,
                context_part_name
            )
            base_dir = os.path.join(
                '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
                folder_name)
            # Path to save results of run
            logdir = os.path.join(RESULTS_PATH, base_dir)
            print('This run directory: %s' % logdir)

            # Create and train model
            model = WaveletBLSTM(params=params, logdir=logdir)
            model.fit_without_validation(data_train, verbose=verbose)
