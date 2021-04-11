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
    folds = 4
    dataset_name = constants.MASS_SS_NAME
    which_expert_list = [1, 2]

    id_try_list = [i for i in range(folds)]
    train_fraction = (folds - 1) / folds

    this_date = datetime.datetime.now().strftime("%Y%m%d")
    for which_expert in which_expert_list:

        # ----- Experiment settings
        experiment_name = 'bsf_update_exp%d' % which_expert
        task_mode = constants.N2_RECORD
        description_str = 'experiments'
        verbose = True

        # Complement experiment folder name with date
        experiment_name = '%s_%s' % (this_date, experiment_name)

        model_version_list = [
            constants.V43
        ]

        # Base parameters
        params = pkeys.default_params.copy()
        # Input border
        params[pkeys.BORDER_DURATION] = 4
        # Conv part: Res-multi-d8
        params[pkeys.BIGGER_CONVOLUTION_PART_OPTION] = 'residual_multi_dilated'
        params[pkeys.BIGGER_STEM_KERNEL_SIZE] = 7
        params[pkeys.BIGGER_STEM_FILTERS] = 64
        params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE] = 3
        params[pkeys.BIGGER_MAX_DILATION] = 8
        params[pkeys.BIGGER_STAGE_1_SIZE] = 1
        params[pkeys.BIGGER_STAGE_2_SIZE] = 1
        params[pkeys.BIGGER_STAGE_3_SIZE] = 0
        # Context part: lstm
        params[pkeys.BIGGER_CONTEXT_PART_OPTION] = 'lstm'
        params[pkeys.BIGGER_LSTM_1_SIZE] = 256
        params[pkeys.BIGGER_LSTM_2_SIZE] = 256
        params[pkeys.FC_UNITS] = 128
        # Training
        params[pkeys.EPOCHS_LR_UPDATE] = 5
        params[pkeys.MAX_LR_UPDATES] = 4

        print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
        dataset = load_dataset(dataset_name, params=params)

        for id_try in id_try_list:
            print('\nUsing validation split %d' % id_try)
            # Generate split
            train_ids, val_ids = utils.split_ids_list_v2(
                dataset.train_ids, split_id=id_try, train_fraction=train_fraction)
            print('Training set IDs:', train_ids)
            data_train = FeederDataset(
                dataset, train_ids, task_mode, which_expert=which_expert)
            print('Validation set IDs:', val_ids)
            data_val = FeederDataset(
                dataset, val_ids, task_mode, which_expert=which_expert)

            for model_version in model_version_list:

                params[pkeys.MODEL_VERSION] = model_version

                folder_name = '%s' % model_version

                base_dir = os.path.join(
                    '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name),
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
