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

    id_try_list = [2]

    # ----- Experiment settings
    experiment_name = 'stat_net_weighted'
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
    model_and_spec_list = [
        (constants.V11_MKD2_STATDOT, 128),  # product dim
        (constants.V11_MKD2_STATDOT, 64),  # product dim
        (constants.V11_MKD2_STATMOD, False),  # modulate logits
        (constants.V11_MKD2_STATMOD, True)  # modulate logits
    ]
    border_duration_list = [
        40,
        20
    ]
    backbone_depth_list = [8]
    backbone_kernel_size_list = [3]
    type_collapse_list = [
        'sigmoid',
        'softmax'
    ]
    params_list = list(itertools.product(
        model_and_spec_list, border_duration_list,
        backbone_depth_list, backbone_kernel_size_list,
        type_collapse_list))

    # Base parameters
    params = pkeys.default_params.copy()

    # Segment net parameters
    params[pkeys.TIME_CONV_MK_PROJECT_FIRST] = False
    params[pkeys.TIME_CONV_MK_DROP_RATE] = 0.0
    params[pkeys.TIME_CONV_MK_SKIPS] = False
    params[pkeys.TIME_CONV_MKD_FILTERS_1] = generate_mkd_specs('none', 3, 64)
    params[pkeys.TIME_CONV_MKD_FILTERS_2] = generate_mkd_specs('dilated', 3, 128)
    params[pkeys.TIME_CONV_MKD_FILTERS_3] = generate_mkd_specs('dilated', 3, 256)
    params[pkeys.FC_UNITS] = 128

    # Stat net parameters
    params[pkeys.STAT_NET_CONV_TYPE_POOL] = constants.MAXPOOL
    params[pkeys.STAT_NET_CONV_INITIAL_FILTERS] = 8
    params[pkeys.STAT_NET_CONV_MAX_FILTERS] = 512
    params[pkeys.STAT_NET_TYPE_BACKBONE] = 'conv'
    params[pkeys.STAT_NET_CONTEXT_DROP_RATE] = 0.2
    params[pkeys.STAT_NET_CONTEXT_DIM] = 128
    params[pkeys.STAT_MOD_NET_BIASED_SCALE] = True
    params[pkeys.STAT_MOD_NET_BIASED_BIAS] = True
    params[pkeys.STAT_MOD_NET_USE_BIAS] = True
    params[pkeys.STAT_DOT_NET_BIASED_KERNEL] = True
    params[pkeys.STAT_DOT_NET_BIASED_BIAS] = True
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

                for model_and_spec, border_duration, backbone_depth, backbone_kernel_size, type_collapse in params_list:

                    model_version = model_and_spec[0]
                    model_config = model_and_spec[1]

                    params[pkeys.MODEL_VERSION] = model_version
                    if model_version == constants.V11_MKD2_STATMOD:
                        params[pkeys.STAT_MOD_NET_MODULATE_LOGITS] = model_config
                    elif model_version == constants.V11_MKD2_STATDOT:
                        params[pkeys.STAT_DOT_NET_PRODUCT_DIM] = model_config
                    else:
                        raise ValueError("unexpected model version %s" % model_version)

                    params[pkeys.BORDER_DURATION] = border_duration
                    params[pkeys.STAT_NET_CONV_DEPTH] = backbone_depth
                    params[pkeys.STAT_NET_CONV_KERNEL_SIZE] = backbone_kernel_size
                    params[pkeys.STAT_NET_TYPE_COLLAPSE] = type_collapse

                    folder_name = '%s_%ds_k%dN%d_%s_spec%d' % (
                        model_version, border_duration, backbone_kernel_size, backbone_depth, type_collapse, model_config)

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
