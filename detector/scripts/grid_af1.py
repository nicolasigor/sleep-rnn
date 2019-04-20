from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import os

import numpy as np

detector_path = '..'
results_folder = 'results'
sys.path.append(detector_path)


if __name__ == '__main__':
    n_to_show = 6
    grid_folder = '20190416_grid_use_log_experimental_train_mass'
    grid_path = os.path.join(detector_path, results_folder, grid_folder)
    grid_list = os.listdir(grid_path)

    grid_name_list = []
    grid_results_list = []
    for grid_setting in grid_list:
        seed_list = os.listdir(os.path.join(grid_path, grid_setting))
        af1_mean_seeds = 0
        for seed in seed_list:
            full_path = os.path.join(grid_path, grid_setting, seed)
            with open(os.path.join(full_path, 'metric.json'), 'r') as file:
                this_dict = json.load(file)
            af1_value = this_dict['val_af1']
            af1_mean_seeds = af1_mean_seeds + af1_value
        af1_mean_seeds = af1_mean_seeds / len(seed_list)
        grid_results_list.append(af1_mean_seeds)
        grid_name_list.append(grid_setting)
    grid_name_list = np.array(grid_name_list)
    grid_results_list = np.array(grid_results_list)
    print('\nNumber of seeds found: %d'
          % len(seed_list))
    print('Showing mean validation performance of Top %d\n' % n_to_show)
    # Sort in ascending order
    idx_sorted = np.argsort(- grid_results_list)

    for idx in idx_sorted[:n_to_show]:
        print('Val AF1 %1.4f for setting %s'
              % (grid_results_list[idx], grid_name_list[idx]))