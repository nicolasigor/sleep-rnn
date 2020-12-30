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


def generate_mk_specs(multi_strategy_name, block_filters):
    if multi_strategy_name == 'linear':
        mk_filters = [
            (3, block_filters // 2),
            (5, block_filters // 4),
            (7, block_filters // 8),
            (9, block_filters // 8),
        ]
    elif multi_strategy_name == 'exp1':
        mk_filters = [
            (3, block_filters // 2),
            (5, block_filters // 4),
            (9, block_filters // 8),
            (17, block_filters // 8),
        ]
    elif multi_strategy_name == 'exp2':
        mk_filters = [
            (3, block_filters // 2),
            (7, block_filters // 4),
            (15, block_filters // 8),
            (31, block_filters // 8),
        ]
    elif multi_strategy_name == 'none':
        mk_filters = [(3, block_filters)]
    else:
        raise ValueError('strategy "%s" invalid' % multi_strategy_name)
    return mk_filters


if __name__ == '__main__':

    id_try_list = [2]

    # ----- Experiment settings
    experiment_name = 'dot_net'
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
    this_date = '20201229'  # datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    # Grid parameters
    model_list = [constants.V11_MKD2_STATDOT]
    stat_drop_rate_list = [
        0.2
    ]
    stat_type_backbone_border_list = [
        ('conv', 40)
    ]
    stat_dot_extra_layer_list = [
        True
    ]
    stat_dot_product_dim_list = [
        64,
        32
    ]
    stat_conv_depth_list = [
        10
    ]
    stat_dot_biased_kernel_list = [
        True,
        False
    ]
    stat_dot_biased_bias_list = [
        False
    ]
    params_list = list(itertools.product(
        model_list,
        stat_drop_rate_list, stat_type_backbone_border_list, stat_dot_extra_layer_list, stat_dot_product_dim_list,
        stat_conv_depth_list, stat_dot_biased_kernel_list, stat_dot_biased_bias_list
        ))

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.TIME_CONV_MK_PROJECT_FIRST] = False
    params[pkeys.TIME_CONV_MK_DROP_RATE] = 0.0
    params[pkeys.TIME_CONV_MK_SKIPS] = False

    # Segment net parameters
    params[pkeys.TIME_CONV_MKD_FILTERS_1] = generate_mkd_specs('none', 3, 64)
    params[pkeys.TIME_CONV_MKD_FILTERS_2] = generate_mkd_specs('dilated', 3, 128)
    params[pkeys.TIME_CONV_MKD_FILTERS_3] = generate_mkd_specs('dilated', 3, 256)

    # stat net parameters
    params[pkeys.STAT_NET_TYPE_POOL] = constants.MAXPOOL
    params[pkeys.STAT_NET_KERNEL_SIZE] = 3
    params[pkeys.STAT_NET_MAX_FILTERS] = 512
    params[pkeys.STAT_NET_OUTPUT_DIM] = 128
    params[pkeys.STAT_DOT_NET_USE_BIAS] = True

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

                for model_version, stat_drop_rate, stat_type_backbone_border, stat_dot_extra_layer, \
                    stat_dot_product_dim, stat_conv_depth, stat_dot_biased_kernel, stat_dot_biased_bias in params_list:

                    if stat_dot_biased_kernel and (stat_dot_product_dim == 64):
                        continue

                    stat_type_backbone = stat_type_backbone_border[0]
                    border_duration = stat_type_backbone_border[1]

                    params[pkeys.MODEL_VERSION] = model_version
                    params[pkeys.STAT_NET_DROP_RATE] = stat_drop_rate
                    params[pkeys.STAT_MOD_NET_TYPE_BACKBONE] = stat_type_backbone
                    params[pkeys.BORDER_DURATION] = border_duration
                    params[pkeys.STAT_NET_DEPTH] = stat_conv_depth
                    params[pkeys.STAT_DOT_NET_BIASED_KERNEL] = stat_dot_biased_kernel
                    params[pkeys.STAT_DOT_NET_BIASED_BIAS] = stat_dot_biased_bias

                    if stat_dot_extra_layer:
                        params[pkeys.FC_UNITS] = 128
                        params[pkeys.STAT_DOT_NET_EXTRA_FC_UNITS] = stat_dot_product_dim
                    else:
                        params[pkeys.FC_UNITS] = stat_dot_product_dim
                        params[pkeys.STAT_DOT_NET_EXTRA_FC_UNITS] = 0

                    folder_name = '%s_dr%1.1f_%s%02d-%ds_extra%d_dotdim%d_bKern%d_bBias%d' % (
                        model_version, stat_drop_rate, stat_type_backbone, stat_conv_depth, border_duration,
                        stat_dot_extra_layer, stat_dot_product_dim, stat_dot_biased_kernel, stat_dot_biased_bias
                    )

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
