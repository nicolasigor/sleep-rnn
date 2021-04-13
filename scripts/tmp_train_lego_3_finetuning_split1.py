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

    id_try_list = [2, 3]
    train_fraction = 0.75
    global_std = None  # 19.209161  # CapFullSS std is 19.209161

    # ----- Experiment settings
    experiment_name = 'lego_3_finetuning'
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1

    description_str = 'experiments'
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_version_list = [
        constants.V43
    ]
    conv_part_list = [
        ('multi-d8', {
            pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'multi_dilated',
            pkeys.BIGGER_STEM_KERNEL_SIZE: 3,
            pkeys.BIGGER_STEM_FILTERS: 64,
            pkeys.BIGGER_MAX_DILATION: 8,
        }),
        ('res-multi-d4', {
            pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual_multi_dilated',
            pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            pkeys.BIGGER_STEM_FILTERS: 32,
            pkeys.BIGGER_MAX_DILATION: 4,
            pkeys.BIGGER_STAGE_1_SIZE: 1,
            pkeys.BIGGER_STAGE_2_SIZE: 1,
            pkeys.BIGGER_STAGE_3_SIZE: 1,
        }),
        ('res-multi-d8', {
            pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual_multi_dilated',
            pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            pkeys.BIGGER_STEM_FILTERS: 64,
            pkeys.BIGGER_MAX_DILATION: 8,
            pkeys.BIGGER_STAGE_1_SIZE: 1,
            pkeys.BIGGER_STAGE_2_SIZE: 1,
            pkeys.BIGGER_STAGE_3_SIZE: 0,
        }),
    ]
    context_part_list = [
        ('lstm', {
            pkeys.BIGGER_CONTEXT_PART_OPTION: 'lstm',
            pkeys.BIGGER_LSTM_1_SIZE: 256,
            pkeys.BIGGER_LSTM_2_SIZE: 256,
            pkeys.FC_UNITS: 128
        })
    ]
    pretraining_option_list = [
        'none1',
        'none2',
        'none3',
    ]
    params_list = list(itertools.product(
        model_version_list, pretraining_option_list, conv_part_list, context_part_list,
    ))

    # Common parameters
    base_params = pkeys.default_params.copy()
    base_params[pkeys.BORDER_DURATION] = 4
    base_params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE] = 3
    base_params[pkeys.EPOCHS_LR_UPDATE] = 5
    base_params[pkeys.MAX_LR_UPDATES] = 4

    print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
    dataset = load_dataset(dataset_name, params=base_params)
    if global_std is not None:
        dataset.global_std = global_std
        print("External global std provided. Dataset now has global std %s" % dataset.global_std)

    # Get training set ids
    all_train_ids = dataset.train_ids
    for id_try in id_try_list:
        print('\nUsing validation split %d' % id_try)
        # Generate split
        train_ids, val_ids = utils.split_ids_list_v2(
            all_train_ids, split_id=id_try, train_fraction=train_fraction)

        print('Training set IDs:', train_ids)
        data_train = FeederDataset(
            dataset, train_ids, task_mode, which_expert=which_expert)
        print('Validation set IDs:', val_ids)
        data_val = FeederDataset(
            dataset, val_ids, task_mode, which_expert=which_expert)

        for model_version, pretraining_option, conv_part, context_part in params_list:

            conv_part_name, conv_part_params = conv_part
            context_part_name, context_part_params = context_part
            params = base_params.copy()
            params.update(conv_part_params)
            params.update(context_part_params)
            params[pkeys.MODEL_VERSION] = model_version

            folder_name = '%s_%s_%s_from-%s' % (
                model_version,
                conv_part_name,
                context_part_name,
                pretraining_option
            )

            base_dir = os.path.join(
                '%s_%s_train_%s' % (
                    experiment_name, task_mode, dataset_name),
                folder_name, 'seed%d' % id_try)

            # Path to save results of run
            logdir = os.path.join(RESULTS_PATH, base_dir)
            print('This run directory: %s' % logdir)

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

            print('Predictions saved at %s' % save_dir)
            print('')
