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

    id_try_list = [0, 1, 2, 3]

    train_fraction = 0.75

    global_std = 19.209161  # CapFullSS std is 19.209161

    # ----- Experiment settings
    experiment_name = 'cap80_finetuning'
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
    model_version_list = [
        constants.V19
    ]
    n_epochs_list = [
        1, 5, 10, 20, 40
    ]
    pretrained_weights_folder_list = [
        ('20210323_cap80_pretrain_exp1_n2_train_cap_full_ss/v19', 'ckpt1'),
        # ('20210323_cap80_pretrain_exp2_n2_train_cap_full_ss/v19', 'ckpt2')
    ]
    params_list = list(itertools.product(
        model_version_list, n_epochs_list, pretrained_weights_folder_list
    ))

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.BORDER_DURATION] = 1

    # Segment net parameters
    params[pkeys.TIME_CONV_MK_PROJECT_FIRST] = False
    params[pkeys.TIME_CONV_MK_DROP_RATE] = 0.0
    params[pkeys.TIME_CONV_MK_SKIPS] = False
    params[pkeys.TIME_CONV_MKD_FILTERS_1] = generate_mkd_specs('none', 3, 64)
    params[pkeys.TIME_CONV_MKD_FILTERS_2] = generate_mkd_specs('dilated', 3, 128)
    params[pkeys.TIME_CONV_MKD_FILTERS_3] = generate_mkd_specs('dilated', 3, 256)
    params[pkeys.FC_UNITS] = 128

    # Fine-tuning settings
    params[pkeys.LEARNING_RATE] = 1e-4
    params[pkeys.FACTOR_INIT_LR_FINE_TUNE] = 0.5
    params[pkeys.EPOCHS_LR_UPDATE] = 100  # Effectively disables the annealing

    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            print('\nModel training on %s_%s' % (dataset_name, task_mode))
            dataset = load_dataset(dataset_name, params=params)
            if global_std is not None:
                dataset.global_std = global_std
                print("External global std provided. Dataset now has global std %s" % dataset.global_std)

            # Get training set ids
            all_train_ids = dataset.train_ids

            for model_version, n_epochs, pretrained_weights_folder in params_list:

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

                    # Set params
                    params[pkeys.MODEL_VERSION] = model_version
                    params[pkeys.MAX_EPOCHS] = n_epochs

                    weight_ckpt_folder = pretrained_weights_folder[0]
                    weight_ckpt_short_name = pretrained_weights_folder[1]

                    folder_name = '%s_epochs%d_from-%s' % (model_version, n_epochs, weight_ckpt_short_name)

                    base_dir = os.path.join(
                        '%s_%s_train_%s' % (
                            experiment_name, task_mode, dataset_name),
                        folder_name, 'seed%d' % id_try)

                    # Path to save results of run
                    logdir = os.path.join(RESULTS_PATH, base_dir)
                    print('This run directory: %s' % logdir)

                    # Create, load pretrained weights, and fine-tune model
                    model = WaveletBLSTM(params=params, logdir=logdir)
                    weight_ckpt_path = os.path.join(RESULTS_PATH, weight_ckpt_folder, 'model', 'ckpt')
                    print('Restoring pretrained weights from %s' % weight_ckpt_path)
                    model.load_checkpoint(weight_ckpt_path)
                    print("Starting fine-tuning")
                    model.fit(data_train, data_val, verbose=verbose, fine_tune=True)

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