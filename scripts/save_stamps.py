from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset, RefactorUnpickler
from sleeprnn.data.inta_ss import NAMES
from sleeprnn.common import constants

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    ckpt_folder = os.path.join('20190506_bsf_n2_train_mass_ss', 'bsf', 'seed0')
    optimal_thr = 0.52
    dataset_name = constants.MASS_SS_NAME
    task_mode = constants.N2_RECORD
    save = False

    # Load data
    dataset = load_dataset(dataset_name)

    ckpt_path = os.path.abspath(os.path.join(
        RESULTS_PATH,
        'predictions_%s' % dataset_name,
        ckpt_folder
    ))
    print('Loading predictions from %s' % ckpt_path)
    print('Using threshold: %1.2f' % optimal_thr)

    set_list = [
        constants.TRAIN_SUBSET,
        constants.VAL_SUBSET,
        constants.TEST_SUBSET]
    stamps_dict = {}

    dataset_name_short = dataset_name.split('_')[0]

    for set_name in set_list:
        filename = os.path.join(
            ckpt_path, 'prediction_%s_%s.pkl' % (task_mode, set_name))
        with open(filename, 'rb') as handle:
            this_pred = RefactorUnpickler(handle).load()
        this_pred.set_probability_threshold(optimal_thr)
        this_ids = this_pred.get_ids()
        for single_id in this_ids:
            print('Processing ID %02d... ' % single_id, end='')
            subject_stamps = this_pred.get_subject_stamps(single_id)
            subject_stamps_time = (subject_stamps / this_pred.fs).astype(np.float32)
            if dataset_name_short == 'mass':
                filename = '%s 01-02-00%02d SpindleTapia.npy' % (dataset_name_short, single_id)
            else:
                filename = '%s %s SpindleTapia.npy' % (dataset_name_short, NAMES[single_id])
            filepath = os.path.join(RESULTS_PATH, filename)
            print('Saved at %s' % filepath)
            if save:
                np.save(filepath, subject_stamps_time)
