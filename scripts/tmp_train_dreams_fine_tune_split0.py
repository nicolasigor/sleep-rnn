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

    # ----- Experiment settings
    experiment_name = 'fine_tune'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.DREAMS_SS_NAME
    ]

    description_str = 'predicting on dreams datasets (small ones) from scratch'
    which_expert = 1
    verbose = True

    # Model grid
    ckpt_list = [
        '20190619_v11_v12_global_n2_train_mass_ss/v11_None_None',
        '20190619_v11_v12_global_n2_train_mass_ss/v12_32_64',
        '20190618_grid_fb_cwtrect_n2_train_mass_ss/fb_0.5',
        '20190617_grid_normalization_n2_train_mass_ss/norm_global',
        '20190618_v18_n2_train_mass_ss/bsf'
    ]

    # Params grid
    factor_fine_tune_list = [1.00, 0.10, 0.01]

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    for factor_fine_tune in factor_fine_tune_list:

        for source_ckpt in ckpt_list:

            ckpt_path_for_params = os.path.abspath(os.path.join(
                RESULTS_PATH, source_ckpt, 'seed0'))

            # Base parameters
            params = pkeys.default_params.copy()
            filename = os.path.join(ckpt_path_for_params, 'params.json')
            with open(filename, 'r') as infile:
                # Overwrite previous defaults with run's params
                params.update(json.load(infile))

            print('\nRestoring params from %s' % ckpt_path_for_params)

            params[pkeys.FACTOR_INIT_LR_FINE_TUNE] = factor_fine_tune

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

                        model_version = params[pkeys.MODEL_VERSION]
                        folder_name = '%s_factor_%s' % (model_version, factor_fine_tune)

                        base_dir = os.path.join(
                            '%s_%s_train_%s' % (
                                experiment_name, task_mode, dataset_name),
                            folder_name, 'seed%d' % id_try)

                        # Path to save results of run
                        logdir = os.path.join(RESULTS_PATH, base_dir)
                        print('This run directory: %s' % logdir)

                        # Create and train model
                        model = WaveletBLSTM(params=params, logdir=logdir)

                        ckpt_path_for_model = os.path.abspath(os.path.join(
                            RESULTS_PATH, source_ckpt, 'seed%d' % id_try))
                        print('Loading weights from %s' % ckpt_path_for_model)
                        model.load_checkpoint(
                            os.path.join(ckpt_path_for_model, 'model', 'ckpt'))

                        # Fine tuning
                        model.fit(data_train, data_val, verbose=verbose, fine_tune=True)

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
