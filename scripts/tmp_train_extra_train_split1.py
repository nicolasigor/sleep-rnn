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

    # ----- Experiment settings
    experiment_name = 'extra_train'
    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]
    which_expert = 1

    description_str = 'v11 with other sleep stages when training'
    verbose = True

    # Complement experiment folder name with date
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = '%s_%s' % (this_date, experiment_name)

    extra_stages_list = [
        ['1'],
        ['R'],
        ['W'],
        ['1', 'W'],
        ['1', 'R'],
        ['W', 'R'],
        ['1', 'W', 'R']
    ]
    invalid_border_stages = ['2', '3', '4']

    # Base parameters
    params = pkeys.default_params.copy()
    params[pkeys.MODEL_VERSION] = constants.V11

    for extra_stages in extra_stages_list:
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

                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # Extra training data
                    print('\nLoading extra training data')
                    print('Extra stages to be added:', extra_stages)
                    print('Borders to be avoided:', invalid_border_stages)
                    border_size = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
                    x_extra, _ = data_train.get_data(
                        augmented_page=True,
                        border_size=border_size,
                        forced_mark_separation_size=0,
                        which_expert=which_expert,
                        pages_subset=constants.WN_RECORD,
                        normalize_clip=True,
                        normalization_mode=task_mode,
                        verbose=False)
                    hypno_train = data_train.get_hypnograms()
                    x_extra_selection = []
                    for s_x, s_h in zip(x_extra, hypno_train):
                        # Look for useful stages:
                        # We do not consider the first, the last,
                        # and the second to last pages
                        useful_pages = []
                        last_page = s_h.size - 1
                        for page in range(1, last_page - 1):
                            this_stage = s_h[page]
                            previous_stage = s_h[page-1]
                            next_stage = s_h[page-1]
                            # the selected page should not be contiguous with N2/N3
                            cr_1 = (this_stage in extra_stages)
                            cr_2 = (previous_stage not in invalid_border_stages)
                            cr_3 = (next_stage not in invalid_border_stages)
                            if cr_1 and cr_2 and cr_3:
                                useful_pages.append(page)
                        useful_pages = np.array(useful_pages)
                        shifted_idx = useful_pages - 1  # WN skips first page
                        s_x_selection = s_x[shifted_idx, :]
                        x_extra_selection.append(s_x_selection)
                    x_extra_selection = np.concatenate(x_extra_selection, axis=0)
                    print('Selected extra training data:', x_extra_selection.shape)
                    # The extra stages should be filled with negative samples
                    # Because they are not N2/N3 and the borders are not N2/N3.
                    y_extra_selection = np.zeros(
                        x_extra_selection.shape, dtype=np.int32)
                    extra_data_train = (x_extra_selection, y_extra_selection)

                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------
                    # -----------------------------------------------

                    model_version = params[pkeys.MODEL_VERSION]
                    extra_stages_str = '-'.join(extra_stages)

                    folder_name = '%s_stages_%s' % (
                        model_version,
                        extra_stages_str
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
                    model.fit(
                        data_train, data_val, verbose=verbose,
                        extra_data_train=extra_data_train)

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
