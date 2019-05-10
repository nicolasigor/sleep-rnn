from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
from pprint import pprint
import sys
import itertools

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import pkeys
from sleeprnn.common import checks

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try_list = [0, 1, 2, 3]

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = '20190510_bsf_aug_rescale_uniform'
    task_mode_list = [
        constants.WN_RECORD
    ]
    dataset_name_list = [
        constants.MASS_SS_NAME
    ]
    which_expert = 1
    verbose = True
    grid_folder_list = None
    # -----

    for dataset_name, task_mode in itertools.product(
            dataset_name_list, task_mode_list):
        print('\nModel predicting on %s_%s' % (dataset_name, task_mode))
        dataset = load_dataset(dataset_name)
        # Get training set ids
        all_train_ids = dataset.train_ids
        # Test data
        test_ids = dataset.test_ids
        if grid_folder_list is None:
            grid_folder_list = os.listdir(os.path.join(
                    RESULTS_PATH,
                    '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name)
                ))
            print('Grid settings found:')
            pprint(grid_folder_list)
        print('')

        for folder_name in grid_folder_list:
            print('\nGrid setting: %s' % folder_name)
            af1_list = []
            for k in id_try_list:
                print('')
                ckpt_path = os.path.abspath(os.path.join(
                    RESULTS_PATH,
                    '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name),
                    '%s' % folder_name,
                    'seed%d' % k
                ))

                # Restore params of ckpt
                params = pkeys.default_params.copy()
                filename = os.path.join(ckpt_path, 'params.json')
                with open(filename, 'r') as infile:
                    # Overwrite previous defaults with run's params
                    params.update(json.load(infile))
                print('Restoring from %s' % ckpt_path)

                # Restore seed
                filename = os.path.join(ckpt_path, 'metric.json')
                with open(filename, 'r') as infile:
                    metric_dict = json.load(infile)
                    this_seed = metric_dict['val_seed']
                    print('Validation split seed: %d' % this_seed)
                    this_af1 = metric_dict['val_af1']
                    af1_list.append(this_af1)
                    this_task_mode = metric_dict['task_mode']
                    if task_mode != this_task_mode:
                        raise ValueError(
                            'Task_mode provided (%s) does not match task mode '
                            'reported in checkpoint (%s)'
                            % (task_mode, this_task_mode))

                # Split to form validation set
                train_ids, val_ids = utils.split_ids_list(
                    all_train_ids, seed=this_seed)
                print('Training set IDs:', train_ids)
                print('Validation set IDs:', val_ids)

                # Create model
                print('Restoring model')
                model = WaveletBLSTM(
                    params,
                    logdir=os.path.join(RESULTS_PATH, 'demo_predict'))
                # Load checkpoint
                model.load_checkpoint(
                    os.path.join(ckpt_path, 'model', 'ckpt'))

                # Save path for predictions
                save_dir = os.path.abspath(os.path.join(
                    RESULTS_PATH,
                    'predictions_%s' % dataset_name,
                    '%s_%s_train_%s' % (ckpt_folder, task_mode, dataset_name),
                    '%s' % folder_name,
                    'seed%d' % k
                ))
                checks.ensure_directory(save_dir)

                # Predict
                ids_dict = {
                    constants.TRAIN_SUBSET: train_ids,
                    constants.VAL_SUBSET: val_ids,
                    constants.TEST_SUBSET: test_ids
                }
                for set_name in ids_dict.keys():
                    print('Predicting %s' % set_name, flush=True)
                    data_inference = FeederDataset(
                        dataset, ids_dict[set_name], task_mode,
                        which_expert=which_expert)
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
            mean_af1 = np.mean(af1_list)
            std_af1 = np.std(af1_list)
            print('Val-AF1 List:', af1_list)
            print('Mean: %1.4f' % mean_af1)
            print('Std: %1.4f' % std_af1)
