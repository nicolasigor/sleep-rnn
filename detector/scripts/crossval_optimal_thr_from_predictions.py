from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import pprint

import numpy as np

detector_path = '..'
results_path = os.path.join(detector_path, 'results')
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA
from sleep.mass_k import MASSK
from evaluation import data_manipulation
from evaluation import metrics
from sleep import postprocessing
from utils import param_keys
from utils import constants
from utils import errors


def get_border_size(my_p):
    border_duration = my_p[param_keys.BORDER_DURATION]
    fs = my_p[param_keys.FS]
    border_size = fs * border_duration
    return border_size


if __name__ == '__main__':

    # Set checkpoint from where to restore, relative to results
    seed_list = [123, 234, 345, 456]
    ckpt_folder = '20190413_bsf_ss_using_angle'
    grid_folder_list = None
    whole_night = False
    dataset_name = constants.MASS_NAME
    verbose = False
    # Performance settings
    res_thr = 0.02
    start_thr = 0.3
    end_thr = 0.6
    # Post processing settings
    min_separation = 0.3
    min_duration = 0.2
    max_duration = 4.0

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    errors.check_valid_value(
        dataset_name, 'dataset_name',
        [constants.MASS_NAME, constants.INTA_NAME, constants.MASSK_NAME])
    if dataset_name == constants.MASS_NAME:
        dataset = MASS(load_checkpoint=True)
    elif dataset_name == constants.INTA_NAME:
        dataset = INTA(load_checkpoint=True)
    else:
        dataset = MASSK(load_checkpoint=True)

    # Get training set ids
    print('Loading train set... ', end='', flush=True)
    all_train_ids = dataset.train_ids
    # Get subjects data, with the expert used for training
    all_x, all_y = dataset.get_subset_data(all_train_ids, which_expert=1,
                                           verbose=verbose)
    print('Signals and marks loaded... ', end='', flush=True)
    all_pages = dataset.get_subset_pages(all_train_ids, verbose=verbose)
    print('Pages loaded.', flush=True)

    # Prepare expert labels into marks
    print('Preparing labels... ', end='', flush=True)
    all_y_stamps = postprocessing.generate_mark_intervals_with_list(
        all_y, all_pages, 200, 200, thr=None, postprocess=False)
    print('Done')

    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            results_path,
            'predictions_%s' % dataset_name,
            '%s_train_%s' % (ckpt_folder, dataset_name)
        ))
        print('Grid settings found:')
        pprint.pprint(grid_folder_list)

    print('')

    # Load predictions (probability vectors for each page),
    # 200/factor resolution (default factor 8)
    set_list = ['val']
    y_pred = {}
    n_seeds = len(seed_list)
    if whole_night:
        descriptor = '_whole_night_'
    else:
        descriptor = '_'
    for j, folder_name in enumerate(grid_folder_list):
        print('\nGrid setting: %s' % folder_name)
        y_pred[folder_name] = []
        for k in range(n_seeds):
            print('\n%d / %d' % (k+1, n_seeds))
            if j == 0:
                this_seed = seed_list[k]
                print('Validation split seed: %d' % this_seed)
            # Restore predictions
            ckpt_path = os.path.abspath(os.path.join(
                results_path,
                'predictions_%s' % dataset_name,
                '%s_train_%s' % (ckpt_folder, dataset_name),
                folder_name,
                'seed%d' % k
            ))
            print('Loading predictions from %s' % ckpt_path)
            this_dict = {}
            for set_name in set_list:
                this_dict[set_name] = np.load(
                    os.path.join(ckpt_path, 'y_pred%s%s.npy'
                                 % (descriptor, set_name)),
                    allow_pickle=True)
                # Keep only class 1 probability
                this_dict[set_name] = [this_y_pred[..., 1]
                                       for this_y_pred in this_dict[set_name]]
            y_pred[folder_name].append(this_dict)
    print('\nDone')

    # Adjust thr
    n_thr = int(np.round((end_thr - start_thr) / res_thr + 1))
    thr_list = [start_thr + res_thr * i for i in range(n_thr)]
    # print(thr_list)
    print('Number of thresholds to be evaluated: %d' % len(thr_list))

    # ---------------- Compute performance
    crossval_af1_mean = {}
    crossval_af1_std = {}
    val_f1 = {}
    first_iou = 0
    last_iou = 1
    res_iou = 0.01
    n_points = int(np.round((last_iou - first_iou) / res_iou))
    full_iou_list = np.arange(n_points + 1) * res_iou + first_iou
    for folder_name in grid_folder_list:
        print('\nGrid setting: %s' % folder_name)
        crossval_af1_mean[folder_name] = []
        crossval_af1_std[folder_name] = []
        for thr in thr_list:
            print('\nUsing thr %1.4f' % thr)
            val_f1[folder_name] = []
            for k, seed in enumerate(seed_list):
                # Prepare expert labels
                _, val_ids = data_manipulation.split_ids_list(
                    all_train_ids, seed=seed)
                print(val_ids)
                val_idx = [all_train_ids.index(this_id) for this_id in val_ids]
                y_thr = [all_y_stamps[i] for i in val_idx]
                pages = [all_pages[i] for i in val_idx]
                # Prepare model predictions
                print('Preparing predictions', flush=True)
                y_pred_thr = postprocessing.generate_mark_intervals_with_list(
                    y_pred[folder_name][k]['val'], pages, 200 // 8, 200,
                    thr=thr, min_separation=min_separation,
                    min_duration=min_duration, max_duration=max_duration)
                # Go through several IoU values
                print('Computing F1 Curve... ', flush=True, end='')
                all_f1_list = [metrics.f1_vs_iou(
                    this_y, this_y_pred, full_iou_list)
                               for (this_y, this_y_pred)
                               in zip(y_thr, y_pred_thr)]
                all_f1_list = np.stack(all_f1_list, axis=1)
                mean_f1 = np.mean(all_f1_list, axis=1)
                # For AF1, we need first and last value to be halved
                mean_f1[0] = mean_f1[0] / 2
                mean_f1[-1] = mean_f1[-1] / 2
                val_f1[folder_name].append(mean_f1)
                print('Ready', flush=True)
            val_af1 = np.stack(val_f1[folder_name], axis=1).mean(axis=0)
            crossval_af1_mean[folder_name].append(val_af1.mean())
            crossval_af1_std[folder_name].append(val_af1.std())
        print('Done')

    # Search optimum
    print('')
    for j, folder_name in enumerate(grid_folder_list):
        max_idx = np.argmax(np.array(crossval_af1_mean[folder_name]))
        print('%s: Optimum at %1.4f with value %1.4f +- %1.4f' % (
            folder_name,
            thr_list[max_idx],
            crossval_af1_mean[folder_name][max_idx],
            crossval_af1_std[folder_name][max_idx]))
