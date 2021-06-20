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
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT

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
    def predict_dataset(
            self, data_inference,
            signal_transform_fn=None,  # For perturbation experiments
            time_reverse=False,  # For temporal inversion experiment
    ):
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


def get_opt_thr_str(optimal_thr_list, ckpt_folder, grid_folder):
    seeds_best_thr_string = ', '.join(['%1.2f' % thr for thr in optimal_thr_list])
    str_to_register = "    os.path.join('%s', '%s'): [%s]," % (ckpt_folder, grid_folder, seeds_best_thr_string)
    return str_to_register


def remove_band(signal, fs, lowcut, highcut):
    if lowcut == 0:
        lowcut = None
    if highcut >= 30:
        highcut = None
    band_signal = utils.apply_bandpass(signal, fs, lowcut, highcut)
    signal_without_band = signal - band_signal
    return signal_without_band


if __name__ == '__main__':
    task_mode = constants.N2_RECORD
    this_date = '20210620'

    configs = [
        # dict(
        #     dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=1,
        #     ckpt_folder="20210411_5fold-cv_exp1_n2_train_moda_ss"),
        dict(
            dataset_name=constants.MASS_KC_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_mass_kc"),
        dict(
            dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_moda_ss"),
    ]

    # Perturbations
    scale_list = np.arange(0.5, 1.5 + 0.001, 0.1)
    perturbations_scale = [
        ('scale-%1.1f' % scale, dict(signal_transform_fn=lambda x: scale * x, time_reverse=False))
        for scale in scale_list]
    perturbations_inversion = [
        ('invert-value', dict(signal_transform_fn=lambda x: -x, time_reverse=False)),
        ('invert-time', dict(signal_transform_fn=lambda x: x, time_reverse=True)),
    ]
    band_list = [(0, 2), (2, 4), (4, 8), (8, 11), (10, 16), (16, 30)]
    perturbations_filter = [
        (
            'filter-%d-%d' % (b[0], b[1]),
            dict(signal_transform_fn=lambda x: remove_band(x, fs=200, lowcut=b[0], highcut=b[1]), time_reverse=False)
        ) for b in band_list
    ]
    perturbations = perturbations_scale + perturbations_inversion + perturbations_filter
    # pprint(perturbations)

    configs_std = get_configs_std(configs)
    # example: fold_std = configs_std[hash_name][fold_id]
    for config in configs:
        print("\nPerturbations on dataset %s-e%d" % (config["dataset_name"], config["which_expert"]))
        dataset = load_dataset(config["dataset_name"], verbose=False)
        partitions = get_partitions(dataset, config["strategy"], config["n_seeds"])
        # example: ids = partitions[set_name][fold_id]
        experiment_name = '%s_from_%s_desc_perturbation_to' % (this_date, config["ckpt_folder"])
        experiment_name_full = '%s_e%d_%s_train_%s' % (
            experiment_name, config["which_expert"], task_mode, config["dataset_name"])

        grid_folder_list = os.listdir(os.path.join(RESULTS_PATH, config["ckpt_folder"]))
        for grid_folder in grid_folder_list:
            print("\nProcessing grid folder %s" % grid_folder)
            grid_folder_complete = os.path.join(config["ckpt_folder"], grid_folder)

            for fold_id in range(len(partitions['test'])):
                print("    Processing fold %d. " % fold_id, end='')
                # Set appropriate global STD
                dataset.global_std = configs_std[get_hash_name(config)][fold_id]
                # Create data feeders
                data_test = FeederDataset(
                    dataset, partitions['test'][fold_id], task_mode, which_expert=config['which_expert'])
                # Retrieve appropriate model
                source_path = os.path.join(RESULTS_PATH, grid_folder_complete, 'fold%d' % fold_id)
                model = get_model(source_path, config["dataset_name"])

                for perturbation_name, perturbation_args in perturbations:
                    print("        Processing perturbation %s" % perturbation_name)
                    # --------------  Predict
                    # Save path for predictions
                    perturbation_grid_folder = '%s_%s' % (grid_folder, perturbation_name)
                    base_dir = os.path.join(
                        RESULTS_PATH, experiment_name_full,
                        perturbation_grid_folder,
                        'fold%d' % fold_id
                    )
                    save_dir = os.path.abspath(os.path.join(
                        RESULTS_PATH, 'predictions_%s' % config["dataset_name"], base_dir))
                    checks.ensure_directory(save_dir)
                    feeders_dict = {
                        constants.TEST_SUBSET: data_test
                    }
                    for set_name in feeders_dict.keys():
                        print('        Predicting %s' % set_name, flush=True)
                        data_inference = feeders_dict[set_name]
                        prediction = model.predict_dataset(
                            data_inference, **perturbation_args)
                        filename = os.path.join(
                            save_dir, 'prediction_%s_%s.pkl' % (task_mode, set_name))
                        with open(filename, 'wb') as handle:
                            pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print('        Predictions saved at %s' % save_dir)
                    print('')

    # Print perturbation ckpt thresholds
    print("\n\nTo register in optimal thresholds:")
    for config in configs:
        experiment_name = '%s_from_%s_desc_perturbation_to' % (this_date, config["ckpt_folder"])
        experiment_name_full = '%s_e%d_%s_train_%s' % (
            experiment_name, config["which_expert"], task_mode, config["dataset_name"])
        grid_folder_list = os.listdir(os.path.join(RESULTS_PATH, config["ckpt_folder"]))
        for grid_folder in grid_folder_list:
            grid_folder_complete = os.path.join(config["ckpt_folder"], grid_folder)
            optimal_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[grid_folder_complete]
            for perturbation_name, _ in perturbations:
                perturbation_grid_folder = '%s_%s' % (grid_folder, perturbation_name)
                opt_thr_str = get_opt_thr_str(optimal_thr_list, experiment_name_full, perturbation_grid_folder)
                print(opt_thr_str)
