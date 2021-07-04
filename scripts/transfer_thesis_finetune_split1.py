from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
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
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection.predicted_dataset import PredictedDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


def get_partitions(dataset, strategy, n_seeds):
    train_ids_list = []
    val_ids_list = []
    test_ids_list = []
    if strategy == 'fixed':
        for fold_id in range(n_seeds):
            train_ids, val_ids = utils.split_ids_list_v2(dataset.train_ids, split_id=fold_id)
            train_ids_list.append(train_ids)
            val_ids_list.append(val_ids)
            test_ids_list.append(dataset.test_ids)
    elif strategy == '5cv':
        for cv_seed in range(n_seeds):
            for fold_id in range(5):
                train_ids, val_ids, test_ids = dataset.cv_split(5, fold_id, cv_seed)
                train_ids_list.append(train_ids)
                val_ids_list.append(val_ids)
                test_ids_list.append(test_ids)
    else:
        raise ValueError
    partitions = {'train': train_ids_list, 'val': val_ids_list, 'test': test_ids_list}
    return partitions


def get_standard_deviations(dataset, partitions):
    standard_deviations = {}
    n_folds = len(partitions['train'])
    for fold_id in range(n_folds):
        train_ids = partitions['train'][fold_id]
        val_ids = partitions['val'][fold_id]
        fold_global_std = dataset.compute_global_std(np.concatenate([train_ids, val_ids]))
        standard_deviations[fold_id] = fold_global_std
    return standard_deviations


def get_hash_name(config):
    return '%s_e%d' % (config["dataset_name"], config["which_expert"])


def get_parameters(train_path):
    fname = os.path.join(train_path, 'params.json')
    with open(fname, 'r') as handle:
        loaded_params = json.load(handle)
    params = copy.deepcopy(pkeys.default_params)
    params.update(loaded_params)
    return params


def get_model(train_path, target_dataset_name, logdir=None, overwrite_params=None):
    weight_ckpt_path = os.path.join(train_path, 'model', 'ckpt')
    print('Restoring weights from %s' % weight_ckpt_path)

    params = get_parameters(train_path)
    # We need to use the target dataset postprocessing settings
    if target_dataset_name == constants.INTA_SS_NAME:
        params[pkeys.SS_MIN_SEPARATION] = 0.5
        params[pkeys.SS_MIN_DURATION] = 0.5
        params[pkeys.SS_MAX_DURATION] = 5.0
    else:
        params[pkeys.SS_MIN_SEPARATION] = 0.3
        params[pkeys.SS_MIN_DURATION] = 0.3
        params[pkeys.SS_MAX_DURATION] = 3.0

    if logdir is None:
        logdir = os.path.join(RESULTS_PATH, 'tmp')
    if overwrite_params is not None:
        params.update(overwrite_params)

    loaded_model = WaveletBLSTM(params=params, logdir=logdir)
    loaded_model.load_checkpoint(weight_ckpt_path)

    # loaded_model = None

    return loaded_model


def get_configs_std(configs):
    configs_std = {}
    for config in configs:
        dataset = load_dataset(config["dataset_name"], verbose=False)
        partitions = get_partitions(dataset, config["strategy"], config["n_seeds"])
        stds = get_standard_deviations(dataset, partitions)
        configs_std[get_hash_name(config)] = stds
    return configs_std


if __name__ == '__main__':
    task_mode = constants.N2_RECORD
    this_date = '20210703'  # datetime.datetime.now().strftime("%Y%m%d")

    # n2_subsampling_factors_dict = {
    #     constants.MODA_SS_NAME: [15.0, 30.0, 60.1, 100.0],
    #     constants.MASS_SS_NAME: [6.2, 12.5, 25.0, 50.0, 100.0],
    #     constants.INTA_SS_NAME: [11.4, 22.8, 45.5, 100.0],
    # }
    # n2_subsampling_factors_dict = {
    #     constants.MODA_SS_NAME: [12.5, 25.0, 50.0, 100.0],
    #     constants.MASS_SS_NAME: [12.5, 25.0, 50.0, 100.0],
    # }
    n2_subsampling_factors_dict = {
        constants.MODA_SS_NAME: [10, 20, 40, 60, 100],
    }

    target_configs = [
        dict(
            dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_moda_ss"),
        # dict(
        #     dataset_name=constants.MASS_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
        #     ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_mass_ss"),
    ]
    source_configs = [
        # dict(
        #     dataset_name=constants.CAP_SS_NAME, which_expert=1, strategy='5cv', n_seeds=1,
        #     ckpt_folder="20210621_thesis_whole_5cv_e1_n2_train_cap_ss"),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_mass_ss"),
        # dict(
        #     dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
        #     ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_moda_ss"),
    ]

    configs_std = get_configs_std(source_configs)
    # example: fold_std = configs_std[hash_name][fold_id]
    for target_config in target_configs:
        print("\nPredict target dataset %s-e%d" % (
            target_config["dataset_name"], target_config["which_expert"]))
        target_dataset = load_dataset(target_config["dataset_name"], verbose=False)
        target_partitions = get_partitions(target_dataset, target_config["strategy"], target_config["n_seeds"])
        # example: ids = partitions[set_name][fold_id]
        for source_config in source_configs:
            if source_config["ckpt_folder"] == target_config["ckpt_folder"]:
                continue
            source_grid_folder_list = os.listdir(os.path.join(RESULTS_PATH, source_config["ckpt_folder"]))
            for source_grid_folder in source_grid_folder_list:
                for fold_id in range(len(target_partitions['train'])):

                    if source_config["dataset_name"] == constants.CAP_SS_NAME:
                        fold_id_from_source = (fold_id % 5)
                    else:
                        fold_id_from_source = fold_id

                    target_train_sizes = n2_subsampling_factors_dict[target_config['dataset_name']]
                    for train_size in target_train_sizes:

                        # Set appropriate global STD -- always from source
                        target_dataset.global_std = configs_std[get_hash_name(source_config)][fold_id_from_source]

                        # Create data feeders (test set is always full)
                        data_train = FeederDataset(
                            target_dataset, target_partitions['train'][fold_id],
                            task_mode, which_expert=target_config['which_expert'],
                            n2_subsampling_factor=(train_size / 100))
                        data_val = FeederDataset(
                            target_dataset, target_partitions['val'][fold_id],
                            task_mode, which_expert=target_config['which_expert'],
                            n2_subsampling_factor=(train_size / 100))
                        data_test = FeederDataset(
                            target_dataset, target_partitions['test'][fold_id],
                            task_mode, which_expert=target_config['which_expert'])

                        # Experiment folder
                        experiment_name = '%s_from_%s_desc_finetune_to' % (this_date, source_config["ckpt_folder"])
                        experiment_name_full = '%s_e%d_%s_train_%s' % (
                            experiment_name, target_config["which_expert"],
                            task_mode, target_config["dataset_name"])
                        base_dir = os.path.join(
                            experiment_name_full,
                            '%s_signalsize%1.1f' % (source_grid_folder, train_size),
                            'fold%d' % fold_id)
                        # Path to save results of run
                        logdir = os.path.join(RESULTS_PATH, base_dir)
                        print('This run directory: %s' % logdir)

                        # Overwrite some params
                        overwrite_params = {
                            pkeys.FACTOR_INIT_LR_FINE_TUNE: 0.5,
                            pkeys.MAX_LR_UPDATES: 3,
                            pkeys.VALIDATION_AVERAGE_MODE: constants.MICRO_AVERAGE
                        }

                        # Retrieve appropriate model
                        source_path = os.path.join(
                            RESULTS_PATH, source_config["ckpt_folder"],
                            source_grid_folder, 'fold%d' % fold_id_from_source)
                        model = get_model(
                            source_path, target_config["dataset_name"],
                            logdir=logdir, overwrite_params=overwrite_params)

                        # Fine-tune
                        model.fit(data_train, data_val, verbose=True, fine_tune=True)

                        # --------------  Predict
                        # Save path for predictions
                        save_dir = os.path.abspath(os.path.join(
                            RESULTS_PATH, 'predictions_%s' % target_config["dataset_name"], base_dir))
                        checks.ensure_directory(save_dir)
                        feeders_dict = {
                            constants.TRAIN_SUBSET: data_train,
                            constants.VAL_SUBSET: data_val,
                            constants.TEST_SUBSET: data_test
                        }
                        for set_name in feeders_dict.keys():
                            print('Predicting %s' % set_name, flush=True)
                            data_inference = feeders_dict[set_name]
                            prediction = model.predict_dataset(data_inference)
                            filename = os.path.join(
                                save_dir,
                                'prediction_%s_%s.pkl' % (task_mode, set_name))
                            with open(filename, 'wb') as handle:
                                pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print('Predictions saved at %s' % save_dir)
                        print('')
