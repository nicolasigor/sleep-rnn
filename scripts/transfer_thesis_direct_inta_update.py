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


def get_model(train_path, target_dataset_name):
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
    weight_ckpt_path = os.path.join(train_path, 'model', 'ckpt')
    print('Restoring weights from %s' % weight_ckpt_path)

    loaded_model = WaveletBLSTM(params=params, logdir=os.path.join(RESULTS_PATH, 'tmp'))
    loaded_model.load_checkpoint(weight_ckpt_path)

    # loaded_model = DummyModel()

    return loaded_model


class DummyModel:
    def predict_dataset(self, data_inference):
        probabilities_dict = {}
        all_ids = data_inference.get_ids()
        for sub_id in all_ids:
            pages = data_inference.get_subject_pages(sub_id, constants.WN_RECORD)
            max_size = (np.max(pages) + 1) * (data_inference.page_size // 8)
            global_sequence = np.zeros(max_size, dtype=np.float16)
            probabilities_dict[sub_id] = global_sequence
        prediction = PredictedDataset(dataset=data_inference, probabilities_dict=probabilities_dict)
        return prediction


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
    normalization_mode = 'source'  # {'source', 'target'}
    this_date = '20210605'  # datetime.datetime.now().strftime("%Y%m%d")

    configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_mass_ss"),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e2_n2_train_mass_ss"),
        dict(
            dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_moda_ss"),
        dict(
            dataset_name=constants.INTA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_inta_ss"),
        # dict(
        #     dataset_name=constants.INTA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=1,
        #     ckpt_folder='20210415_5fold-cv_exp1_n2_train_inta_ss'),
        # dict(
        #     dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=1,
        #     ckpt_folder="20210411_5fold-cv_exp1_n2_train_moda_ss"),

    ]

    configs_std = get_configs_std(configs)
    # example: fold_std = configs_std[hash_name][fold_id]
    for target_config in configs:
        print("\nPredict target dataset %s-e%d" % (
            target_config["dataset_name"], target_config["which_expert"]))
        target_dataset = load_dataset(target_config["dataset_name"], verbose=False)
        target_partitions = get_partitions(target_dataset, target_config["strategy"], target_config["n_seeds"])
        # example: ids = partitions[set_name][fold_id]
        for source_config in configs:

            # only something containing inta
            if (target_config['dataset_name'] != constants.INTA_SS_NAME) and (source_config['dataset_name'] != constants.INTA_SS_NAME):
                continue

            if source_config["ckpt_folder"] == target_config["ckpt_folder"]:
                continue
            source_grid_folder_list = os.listdir(os.path.join(RESULTS_PATH, source_config["ckpt_folder"]))
            for source_grid_folder in source_grid_folder_list:
                for fold_id in range(len(target_partitions['train'])):
                    # Set appropriate global STD
                    possible_global_std = {
                        'source': configs_std[get_hash_name(source_config)][fold_id],
                        'target': configs_std[get_hash_name(target_config)][fold_id]}
                    target_dataset.global_std = possible_global_std[normalization_mode]
                    # Create data feeders
                    data_train = FeederDataset(
                        target_dataset, target_partitions['train'][fold_id],
                        task_mode, which_expert=target_config['which_expert'])
                    data_val = FeederDataset(
                        target_dataset, target_partitions['val'][fold_id],
                        task_mode, which_expert=target_config['which_expert'])
                    data_test = FeederDataset(
                        target_dataset, target_partitions['test'][fold_id],
                        task_mode, which_expert=target_config['which_expert'])
                    # Retrieve appropriate model
                    source_path = os.path.join(
                        RESULTS_PATH, source_config["ckpt_folder"],
                        source_grid_folder, 'fold%d' % fold_id)
                    model = get_model(source_path, target_config["dataset_name"])
                    # --------------  Predict
                    # Save path for predictions
                    experiment_name = '%s_from_%s_desc_%sstd_to' % (
                        this_date, source_config["ckpt_folder"], normalization_mode)
                    experiment_name_full = '%s_e%d_%s_train_%s' % (
                        experiment_name, target_config["which_expert"],
                        task_mode, target_config["dataset_name"])
                    base_dir = os.path.join(
                        experiment_name_full, source_grid_folder, 'fold%d' % fold_id)
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
