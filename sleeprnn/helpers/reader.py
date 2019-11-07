import os
import pickle

import numpy as np
import pyedflib

from sleeprnn.common import checks, constants
from sleeprnn.data.dreams_kc import DreamsKC
from sleeprnn.data.dreams_ss import DreamsSS
from sleeprnn.data.inta_ss import IntaSS
from sleeprnn.data.mass_kc import MassKC
from sleeprnn.data.mass_ss import MassSS

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
BASELINES_PATH = os.path.join(PROJECT_ROOT, 'resources', 'comparison_data', 'baselines')
EXPERT_PATH = os.path.join(PROJECT_ROOT, 'resources', 'comparison_data', 'expert')


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
        ckpt_folder,
        dataset_name,
        task_mode,
        seed_id_list,
        set_list=None,
        verbose=True
):
    if verbose:
        print('Loading predictions')
    if set_list is None:
        set_list = [
            constants.TRAIN_SUBSET,
            constants.VAL_SUBSET,
            constants.TEST_SUBSET]
    predictions_dict = {}
    for k in seed_id_list:
        # Restore predictions
        ckpt_path = os.path.abspath(os.path.join(
            RESULTS_PATH,
            'predictions_%s' % dataset_name,
            ckpt_folder,
            'seed%d' % k
        ))
        this_dict = {}
        for set_name in set_list:
            filename = os.path.join(
                ckpt_path,
                'prediction_%s_%s.pkl' % (task_mode, set_name))
            with open(filename, 'rb') as handle:
                this_pred = RefactorUnpickler(handle).load()
            this_dict[set_name] = this_pred
        predictions_dict[k] = this_dict
        if verbose:
            print('Loaded %s' % ckpt_path)
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


def load_dataset(dataset_name, load_checkpoint=True, params=None, verbose=True):
    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [
            constants.MASS_KC_NAME,
            constants.MASS_SS_NAME,
            constants.INTA_SS_NAME,
            constants.DREAMS_KC_NAME,
            constants.DREAMS_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(
            load_checkpoint=load_checkpoint, params=params, verbose=verbose)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(
            load_checkpoint=load_checkpoint, params=params, verbose=verbose)
    elif dataset_name == constants.INTA_SS_NAME:
        dataset = IntaSS(
            load_checkpoint=load_checkpoint, params=params, verbose=verbose)
    elif dataset_name == constants.DREAMS_SS_NAME:
        dataset = DreamsSS(
            load_checkpoint=load_checkpoint, params=params, verbose=verbose)
    else:
        dataset = DreamsKC(
            load_checkpoint=load_checkpoint, params=params, verbose=verbose)
    return dataset


def load_baselines(
        baselines_to_load,
        subject_ids,
        dataset_name,
        which_expert,
        n_folds=10,
):
    baselines_data_dict = {}
    for baseline_name in baselines_to_load:
        # Check if we have results for this baseline
        folder_to_check = os.path.join(
            BASELINES_PATH, baseline_name, dataset_name, 'e%d' % which_expert)
        if os.path.exists(folder_to_check):
            print('%s found. ' % baseline_name, end='', flush=True)
            iou_axis_added = False
            iou_bins_added = False
            tmp_f1_baseline = []
            tmp_recall_baseline = []
            tmp_precision_baseline = []
            tmp_iou_hist_baseline = []
            tmp_iou_mean_baseline = []
            tmp_af1_mean_baseline = []
            right_prefix = '%s_%s_e%d' % (baseline_name, dataset_name, which_expert)
            for k in np.arange(n_folds):
                print(' %d ' % k, end='', flush=True)
                f1_seed = []
                rec_seed = []
                pre_seed = []
                iou_hist_seed = []
                iou_mean_seed = []
                af1_mean_seed = []
                for i, subject_id in enumerate(subject_ids):
                    fname = '%s_fold%d_s%02d.npz' % (
                    right_prefix, k, subject_id)
                    fname_path = os.path.join(
                        folder_to_check, 'fold%d' % k, fname)
                    this_data = np.load(fname_path)
                    if not iou_axis_added:
                        tmp_iou_axis = this_data['iou_axis']
                        iou_axis_added = True
                    if not iou_bins_added:
                        tmp_iou_bins = this_data['iou_hist_bins']
                        iou_bins_added = True
                    f1_seed.append(this_data['f1_vs_iou'])
                    rec_seed.append(this_data['recall_vs_iou'])
                    pre_seed.append(this_data['precision_vs_iou'])
                    iou_hist_seed.append(this_data['iou_hist_values'])
                    iou_mean_seed.append(this_data['subject_iou'])
                    af1_mean_seed.append(this_data['subject_af1'])
                f1_seed = np.stack(f1_seed, axis=0).mean(axis=0)
                rec_seed = np.stack(rec_seed, axis=0).mean(axis=0)
                pre_seed = np.stack(pre_seed, axis=0).mean(axis=0)
                iou_hist_seed = np.stack(iou_hist_seed, axis=0).mean(axis=0)
                iou_mean_seed = np.stack(iou_mean_seed, axis=0).mean(axis=0)
                af1_mean_seed = np.stack(af1_mean_seed, axis=0).mean(axis=0)
                tmp_f1_baseline.append(f1_seed)
                tmp_recall_baseline.append(rec_seed)
                tmp_precision_baseline.append(pre_seed)
                tmp_iou_hist_baseline.append(iou_hist_seed)
                tmp_iou_mean_baseline.append(iou_mean_seed)
                tmp_af1_mean_baseline.append(af1_mean_seed)
            tmp_f1_baseline = np.stack(tmp_f1_baseline, axis=0)
            tmp_recall_baseline = np.stack(tmp_recall_baseline, axis=0)
            tmp_precision_baseline = np.stack(tmp_precision_baseline, axis=0)
            tmp_iou_hist_baseline = np.stack(tmp_iou_hist_baseline, axis=0)
            tmp_iou_mean_baseline = np.stack(tmp_iou_mean_baseline, axis=0)
            tmp_af1_mean_baseline = np.stack(tmp_af1_mean_baseline, axis=0)
            baselines_data_dict[baseline_name] = {
                constants.F1_VS_IOU: tmp_f1_baseline,
                constants.RECALL_VS_IOU: tmp_recall_baseline,
                constants.PRECISION_VS_IOU: tmp_precision_baseline,
                constants.IOU_HIST_BINS: tmp_iou_bins,
                constants.IOU_CURVE_AXIS: tmp_iou_axis,
                constants.IOU_HIST_VALUES: tmp_iou_hist_baseline,
                constants.MEAN_IOU: tmp_iou_mean_baseline,
                constants.MEAN_AF1: tmp_af1_mean_baseline
            }
            print('Loaded.')
        else:
            print('%s not found.' % baseline_name)
            baselines_data_dict[baseline_name] = None
    return baselines_data_dict


def load_ss_expert_performance():
    expert_f1_curve_mean = np.loadtxt(
        os.path.join(EXPERT_PATH, 'ss_f1_vs_iou_expert_mean.csv'), delimiter=',')
    expert_f1_curve_std = np.loadtxt(
        os.path.join(EXPERT_PATH, 'ss_f1_vs_iou_expert_std.csv'), delimiter=',')
    expert_f1_curve_mean = expert_f1_curve_mean[1:, :]
    expert_f1_curve_std = expert_f1_curve_std[1:, :]
    expert_rec_prec = np.loadtxt(
        os.path.join(EXPERT_PATH, 'ss_pr_expert_mean.csv'), delimiter=',')
    expert_recall = expert_rec_prec[0]
    expert_precision = expert_rec_prec[1]
    expert_data_dict = {
        constants.IOU_CURVE_AXIS: expert_f1_curve_mean[:, 0],
        '%s_mean' % constants.F1_VS_IOU: expert_f1_curve_mean[:, 1],
        '%s_std' % constants.F1_VS_IOU: expert_f1_curve_std[:, 1],
        constants.RECALL: expert_recall,
        constants.PRECISION: expert_precision
    }
    return expert_data_dict
