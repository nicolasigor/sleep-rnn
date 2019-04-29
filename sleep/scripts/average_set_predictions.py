from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

detector_path = '..'
results_path = os.path.join(detector_path, 'results')
sys.path.append(detector_path)

from utils import constants


if __name__ == '__main__':

    set_list = ['alltrain', 'test']

    # Set paths for single-run predictions

    # ['True_experimental', 'True_v3-ff', 'False_v3-ff', 'False_experimental']
    main_folder_name = os.path.join(
        '20190416_grid_use_log_experimental_train_mass',
        'False_experimental'
    )
    whole_night = True

    dataset_name = constants.MASS_NAME
    npy_load_list = [
        os.path.join(
            main_folder_name, 'seed%d' % i)
        for i in range(4)
    ]
    npy_avg_save_folder = os.path.join(
        main_folder_name, 'avg')
    if whole_night:
        descriptor = '_whole_night_'
    else:
        descriptor = '_'

    # Load predictions
    prediction_folder = 'predictions_%s' % dataset_name
    save_path = os.path.abspath(os.path.join(
        results_path, prediction_folder, npy_avg_save_folder))

    for set_name in set_list:
        print('Averaging predictions for %s set' % set_name)
        pred_list = []
        for npy_folder in npy_load_list:
            this_path = os.path.abspath(os.path.join(
                results_path, prediction_folder, npy_folder))
            print('Loading from %s' % this_path)
            this_pred = np.load(
                os.path.join(
                    this_path, 'y_pred%s%s.npy' % (descriptor, set_name)),
                allow_pickle=True)
            pred_list.append(this_pred)
        set_size = pred_list[0].shape[0]
        n_preds = len(pred_list)
        print('Set size: %d' % set_size)
        avg_pred = []
        for i in range(set_size):
            mean = pred_list[0][i]
            for j in range(1, n_preds):
                mean = mean + pred_list[j][i]
            mean = mean / n_preds
            avg_pred.append(mean)

        # Save average
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(
            os.path.join(
                save_path, 'y_pred%s%s.npy' % (descriptor, set_name)),
            avg_pred)
    print('Predictions saved at %s' % save_path)
