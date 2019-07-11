import os
import pickle

import numpy as np
import pyedflib

from sleeprnn.common import constants

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')


def replace_submodule_in_module(module, old_submodule, new_submodule):
    module_splitted = module.split(".")
    if old_submodule in module_splitted:
        idx_name = module_splitted.index(old_submodule)
        module_splitted[idx_name] = new_submodule
    new_module = ".".join(module_splitted)
    return new_module


class RefactorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = replace_submodule_in_module(module, 'sleep', 'sleeprnn')
        module = replace_submodule_in_module(module, 'neuralnet', 'nn')
        return super().find_class(module, name)


def read_prediction_with_seeds(
        ckpt_folder, dataset_name, task_mode, set_list=None, verbose=True):

    ckpt_path = os.path.abspath(os.path.join(
            RESULTS_PATH, 'predictions_%s' % dataset_name,
            ckpt_folder))
    if verbose:
        print('Loading predictions from %s' % ckpt_path)
    seed_folders = os.listdir(ckpt_path)
    seed_id_list = [int(seed_name[-1]) for seed_name in seed_folders]
    seed_id_list.sort()

    if set_list is None:
        set_list = [
            constants.TRAIN_SUBSET,
            constants.VAL_SUBSET,
            constants.TEST_SUBSET]

    predictions_dict = {}
    for k in seed_id_list:
        seed_path = os.path.join(ckpt_path, 'seed%d' % k)
        this_dict = {}
        for set_name in set_list:
            filename = os.path.join(
                seed_path, 'prediction_%s_%s.pkl' % (task_mode, set_name))
            with open(filename, 'rb') as handle:
                this_pred = RefactorUnpickler(handle).load()
            this_dict[set_name] = this_pred
        predictions_dict[k] = this_dict
        if verbose:
            print('Loaded seed %d / %d' % (k+1, len(seed_id_list)))
    return predictions_dict


def read_signals_from_edf(filepath):
    signal_dict = {}
    with pyedflib.EdfReader(filepath) as file:
        signal_names = file.getSignalLabels()
        for k, name in enumerate(signal_names):
            this_signal = file.readSignal(k)
            signal_dict[name] = this_signal
    return signal_dict


def load_raw_inta_stamps(
        path_stamps, path_signals, min_samples=20, chn_idx=0, max_samples=1400):
    with pyedflib.EdfReader(path_signals) as file:
        signal = file.readSignal(chn_idx)
        channel_name = file.getLabel(chn_idx)
        print('Using %s channel' % channel_name)
        signal_len = signal.shape[0]
    data = np.loadtxt(path_stamps)
    for_this_channel = (data[:, -1] == chn_idx+1)
    data = data[for_this_channel]
    data = np.round(data).astype(np.int32)

    # Remove zero duration marks, and ensure that start time < end time
    new_data = []
    for i in range(data.shape[0]):
        if data[i, 0] > data[i, 1]:
            print('End time < Start time fixed')
            aux = data[i, 0]
            data[i, 0] = data[i, 1]
            data[i, 1] = aux
            new_data.append(data[i, :])
        elif data[i, 0] < data[i, 1]:
            new_data.append(data[i, :])
        else:  # Zero duration (equality)
            print('Zero duration stamp found and removed')
    data = np.stack(new_data, axis=0)

    # Remove marks with duration less than min_samples
    new_data = []
    for i in range(data.shape[0]):
        if data[i, 1] - data[i, 0] >= min_samples:
            new_data.append(data[i, :])
        else:
            this_n_samples = data[i, 1] - data[i, 0]
            print('Stamp with too few samples removed (%d)' % this_n_samples)
    data = np.stack(new_data, axis=0)

    # Remove marks with duration greater than max_samples
    new_data = []
    for i in range(data.shape[0]):
        if data[i, 1] - data[i, 0] <= max_samples:
            new_data.append(data[i, :])
        else:
            this_n_samples = data[i, 1] - data[i, 0]
            print('Stamp with too many samples removed (%d)' % this_n_samples)
    data = np.stack(new_data, axis=0)

    # Remove stamps outside signal boundaries
    new_data = []
    for i in range(data.shape[0]):
        if data[i, 1] < signal_len:
            new_data.append(data[i, :])
        else:
            print('Stamp outside boundaries found and removed')
    data = np.stack(new_data, axis=0)

    raw_stamps = data[:, [0, 1]]
    valid = data[:, 4]
    raw_stamps_1 = raw_stamps[valid == 1]
    raw_stamps_2 = raw_stamps[valid == 2]
    return raw_stamps_1, raw_stamps_2
