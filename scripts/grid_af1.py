from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    n_to_show = 30
    grid_folder = '20190504_bsf_wn_train_mass_kc'
    grid_path = os.path.join(RESULTS_PATH, grid_folder)
    grid_list = os.listdir(grid_path)

    grid_name_list = []
    grid_results_mean_list = []
    grid_results_std_list = []
    for grid_setting in grid_list:
        seed_list = os.listdir(os.path.join(grid_path, grid_setting))
        af1_seeds = []
        for seed in seed_list:
            full_path = os.path.join(grid_path, grid_setting, seed)
            with open(os.path.join(full_path, 'metric.json'), 'r') as file:
                this_dict = json.load(file)
            af1_value = this_dict['val_af1']
            af1_seeds.append(af1_value)
        af1_mean_seeds = np.mean(af1_seeds)
        af1_std_seeds = np.std(af1_seeds)
        grid_results_mean_list.append(af1_mean_seeds)
        grid_results_std_list.append(af1_std_seeds)
        grid_name_list.append(grid_setting)
    grid_name_list = np.array(grid_name_list)
    grid_results_mean_list = np.array(grid_results_mean_list)
    grid_results_std_list = np.array(grid_results_std_list)
    print('\nNumber of seeds found: %d'
          % len(seed_list))
    print('Showing mean validation performance of Top %d\n' % n_to_show)
    # Sort in ascending order
    idx_sorted = np.argsort(- grid_results_mean_list)

    for idx in idx_sorted[:n_to_show]:
        print('Val AF1 %1.4f +- %1.4f for setting %s'
              % (
                  grid_results_mean_list[idx],
                  grid_results_std_list[idx],
                  grid_name_list[idx]))
