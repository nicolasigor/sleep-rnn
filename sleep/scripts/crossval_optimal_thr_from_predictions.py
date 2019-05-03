from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
results_folder = 'results'
sys.path.append(project_root)

from sleep.data.inta_ss import IntaSS
from sleep.data.mass_kc import MassKC
from sleep.data.mass_ss import MassSS
from sleep.data import data_ops, metrics, postprocessing
from sleep.utils import constants
from sleep.utils import checks
from sleep.utils import pkeys

RESULTS_PATH = os.path.join(project_root, 'sleep', 'results')
SEED_LIST = [123, 234, 345, 456]

if __name__ == '__main__':

    # Set checkpoint from where to restore, relative to results
    ckpt_folder = '20190502_bsf_norm_activity'
    grid_folder_list = None
    whole_night = False
    dataset_name = constants.MASS_KC_NAME
    which_expert = 1
    verbose = False

    # Performance settings
    res_thr = 0.02
    start_thr = 0.3
    end_thr = 0.6

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [
            constants.MASS_KC_NAME,
            constants.MASS_SS_NAME,
            constants.INTA_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=True)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(load_checkpoint=True)
    else:
        dataset = IntaSS(load_checkpoint=True)

    # Get training set ids
    print('Loading train set... ', end='', flush=True)
    all_train_ids = dataset.train_ids
    # Get subjects data, with the expert used for training
    print('Signals and marks loaded... ', end='', flush=True)
    all_pages = dataset.get_subset_pages(
        all_train_ids, verbose=verbose, whole_night=whole_night)
    all_wholenight_pages = dataset.get_subset_pages(
        all_train_ids, verbose=verbose, whole_night=True)
    print('Pages loaded.', flush=True)

    # Prepare expert labels into marks
    print('Preparing labels... ', end='', flush=True)
    all_y_stamps = dataset.get_subset_stamps(
        all_train_ids,
        which_expert=which_expert,
        whole_night=whole_night,
        verbose=verbose)
    print('Done')

    if whole_night:
        descriptor = '_whole_night_'
    else:
        descriptor = '_'

    if grid_folder_list is None:
        grid_folder_list = os.listdir(os.path.join(
            RESULTS_PATH,
            '%s%strain_%s' % (ckpt_folder, descriptor, dataset_name)
        ))
        print('Grid settings found:')
        pprint(grid_folder_list)

    print('')

    # Load predictions (probability vectors for each page),
    # 200/factor resolution (default factor 8)
    set_list = [constants.VAL_SUBSET]
    y_pred = {}
    n_seeds = len(SEED_LIST)

    for j, folder_name in enumerate(grid_folder_list):

        print('\nGrid setting: %s' % folder_name)
        y_pred[folder_name] = []
        for k in range(n_seeds):
            print('\n%d / %d' % (k+1, n_seeds))
            if j == 0:
                this_seed = SEED_LIST[k]
                print('Validation split seed: %d' % this_seed)
            # Restore predictions
            ckpt_path = os.path.abspath(os.path.join(
                RESULTS_PATH,
                'predictions_%s' % dataset_name,
                '%s%strain_%s' % (ckpt_folder, descriptor, dataset_name),
                '%s' % folder_name,
                'seed%d' % k
            ))
            print('Loading predictions from %s' % ckpt_path)
            this_dict = {}
            for set_name in set_list:
                this_dict[set_name] = np.load(
                    os.path.join(ckpt_path, 'y_pred%s%s.npy'
                                 % (descriptor, set_name)),
                    allow_pickle=True)
            y_pred[folder_name].append(this_dict)
    print('\nDone')

    # Adjust thr
    n_thr = int(np.round((end_thr - start_thr) / res_thr + 1))
    thr_list = [start_thr + res_thr * i for i in range(n_thr)]
    thr_list = np.asarray(thr_list)
    # print(thr_list)
    print('Number of thresholds to be evaluated: %d' % len(thr_list))

    # ---------------- Compute performance
    params = pkeys.default_params.copy()
    if dataset_name in [constants.MASS_SS_NAME, constants.INTA_SS_NAME]:
        min_separation = params[pkeys.SS_MIN_SEPARATION]
        min_duration = params[pkeys.SS_MIN_DURATION]
        max_duration = params[pkeys.SS_MAX_DURATION]
    else:
        min_separation = params[pkeys.KC_MIN_SEPARATION]
        min_duration = params[pkeys.KC_MIN_DURATION]
        max_duration = params[pkeys.KC_MAX_DURATION]

    crossval_af1_mean = {}
    crossval_af1_std = {}
    for folder_name in grid_folder_list:
        print('\nGrid setting: %s' % folder_name)
        crossval_af1_mean[folder_name] = []
        crossval_af1_std[folder_name] = []
        for thr in thr_list:
            print('Processing thr %1.4f' % thr)
            val_af1 = []
            for k, seed in enumerate(SEED_LIST):
                # Prepare expert labels
                _, val_ids = data_ops.split_ids_list(
                    all_train_ids, seed=seed, verbose=verbose)
                if verbose:
                    print('Val IDs:', val_ids)
                val_idx = [all_train_ids.index(this_id) for this_id in val_ids]
                y_thr = [all_y_stamps[i] for i in val_idx]
                pages = [all_pages[i] for i in val_idx]
                wholenight_pages = [all_wholenight_pages[i] for i in val_idx]

                # Prepare model predictions
                y_pred_thr = postprocessing.generate_mark_intervals_with_list(
                    y_pred[folder_name][k][constants.VAL_SUBSET],
                    wholenight_pages,
                    fs_input=200 // 8,
                    fs_output=200,
                    thr=thr,
                    min_separation=min_separation,
                    min_duration=min_duration,
                    max_duration=max_duration)

                # Go through several IoU values
                if not whole_night:
                    # Keep only N2 stamps
                    y_pred_thr = [data_ops.extract_pages_with_stamps(
                        this_y_pred_thr, this_pages, dataset.page_size
                    ) for (this_y_pred_thr, this_pages)
                        in zip(y_pred_thr, pages)]

                val_af1_at_thr = metrics.average_metric_with_list(
                    y_thr,
                    y_pred_thr,
                    verbose=verbose)

                val_af1.append(val_af1_at_thr)

            crossval_af1_mean[folder_name].append(np.mean(val_af1))
            crossval_af1_std[folder_name].append(np.std(val_af1))
        print('Done')

    # Search optimum
    print('\nReport for %s%strain_%s' % (ckpt_folder, descriptor, dataset_name))
    for j, folder_name in enumerate(grid_folder_list):
        max_idx = np.argmax(np.array(crossval_af1_mean[folder_name])).item()
        half_idx = np.where(np.isclose(thr_list, 0.5))[0].item()

        print('Val AF1 %1.4f +- %1.4f (mu %1.4f). %1.4f +- %1.4f (mu 0.5) for setting %s'
              % (crossval_af1_mean[folder_name][max_idx],
                 crossval_af1_std[folder_name][max_idx],
                 thr_list[max_idx],
                 crossval_af1_mean[folder_name][half_idx],
                 crossval_af1_std[folder_name][half_idx],
                 folder_name))
