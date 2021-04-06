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
    train_fraction = 0.75
    global_std = 19.209161  # CapFullSS std is 19.209161

    # ----- Experiment settings
    experiment_name = 'lego_1_finetuning'
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
        ('res-d1', {
            pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual',
            pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            pkeys.BIGGER_STEM_FILTERS: 32,
            pkeys.BIGGER_MAX_DILATION: 1,
            pkeys.BIGGER_STAGE_1_SIZE: 2,
            pkeys.BIGGER_STAGE_2_SIZE: 2,
            pkeys.BIGGER_STAGE_3_SIZE: 4,
        }),
        ('res-d4', {
            pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual',
            pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            pkeys.BIGGER_STEM_FILTERS: 32,
            pkeys.BIGGER_MAX_DILATION: 4,
            pkeys.BIGGER_STAGE_1_SIZE: 2,
            pkeys.BIGGER_STAGE_2_SIZE: 2,
            pkeys.BIGGER_STAGE_3_SIZE: 3,
        }),
        ('res-d8', {
            pkeys.BIGGER_CONVOLUTION_PART_OPTION: 'residual',
            pkeys.BIGGER_STEM_KERNEL_SIZE: 7,
            pkeys.BIGGER_STEM_FILTERS: 64,
            pkeys.BIGGER_MAX_DILATION: 8,
            pkeys.BIGGER_STAGE_1_SIZE: 2,
            pkeys.BIGGER_STAGE_2_SIZE: 4,
            pkeys.BIGGER_STAGE_3_SIZE: 0,
        }),
    ]

    context_part_list = [
        ('att', {
            pkeys.BIGGER_CONTEXT_PART_OPTION: 'attention',
            pkeys.BIGGER_ATT_N_BLOCKS: 3,
            pkeys.BIGGER_ATT_TYPE_NORM: 'layernorm'
        }),
        # ('lstm', {
        #     pkeys.BIGGER_CONTEXT_PART_OPTION: 'lstm',
        #     pkeys.BIGGER_LSTM_1_SIZE: 256,
        #     pkeys.BIGGER_LSTM_2_SIZE: 256,
        #     pkeys.FC_UNITS: 128
        # }),
        # ('res_lstm_ln', {
        #     pkeys.BIGGER_CONTEXT_PART_OPTION: 'residual_lstm',
        #     pkeys.BIGGER_LSTM_1_SIZE: 256,
        #     pkeys.BIGGER_LSTM_2_SIZE: 256,
        #     pkeys.FC_UNITS: 128,
        #     pkeys.BIGGER_ATT_TYPE_NORM: 'layernorm'
        # }),
        # ('res_lstm_bn', {
        #     pkeys.BIGGER_CONTEXT_PART_OPTION: 'residual_lstm',
        #     pkeys.BIGGER_LSTM_1_SIZE: 256,
        #     pkeys.BIGGER_LSTM_2_SIZE: 256,
        #     pkeys.FC_UNITS: 128,
        #     pkeys.BIGGER_ATT_TYPE_NORM: 'batchnorm'
        # }),
    ]
    pretraining_option_list = [
        'none',
        'ckpt2',
        'ckpt1',
    ]
    params_list = list(itertools.product(
        model_version_list, pretraining_option_list, conv_part_list, context_part_list,
    ))

    # Common parameters
    base_params = pkeys.default_params.copy()
    base_params[pkeys.BORDER_DURATION] = 4
    base_params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE] = 3
    base_params[pkeys.ATT_DROP_RATE] = 0.1
    base_params[pkeys.ATT_DIM] = 256
    base_params[pkeys.ATT_N_HEADS] = 4
    base_params[pkeys.ATT_PE_FACTOR] = 10000

    base_params[pkeys.FACTOR_INIT_LR_FINE_TUNE] = 0.5

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

            pretrain_folder_name = '%s_%s_%s' % (
                model_version,
                conv_part_name,
                context_part_name)
            if pretraining_option == 'none':
                fine_tune = False
                params[pkeys.EPOCHS_LR_UPDATE] = 5
                params[pkeys.MAX_LR_UPDATES] = 4
                weight_ckpt_folder = 'dummy'
            elif pretraining_option == 'ckpt1':
                fine_tune = True
                params[pkeys.EPOCHS_LR_UPDATE] = 4
                params[pkeys.MAX_LR_UPDATES] = 3
                weight_ckpt_folder = os.path.join(
                    '20210404_lego_1_pretrain_exp1_n2_train_cap_full_ss', pretrain_folder_name)
            elif pretraining_option == 'ckpt2':
                fine_tune = True
                params[pkeys.EPOCHS_LR_UPDATE] = 4
                params[pkeys.MAX_LR_UPDATES] = 3
                weight_ckpt_folder = os.path.join(
                    '20210404_lego_1_pretrain_exp2_n2_train_cap_full_ss', pretrain_folder_name)
            else:
                raise ValueError()

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

            # Create, load pretrained weights, and fine-tune model
            if fine_tune:
                weight_ckpt_path = os.path.join(RESULTS_PATH, weight_ckpt_folder, 'model', 'ckpt')
                print('Restoring pretrained weights from %s' % weight_ckpt_path)

            model = WaveletBLSTM(params=params, logdir=logdir)
            if fine_tune:
                model.load_checkpoint(weight_ckpt_path)
                print("Starting fine-tuning")
            model.fit(data_train, data_val, verbose=verbose, fine_tune=fine_tune)

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
