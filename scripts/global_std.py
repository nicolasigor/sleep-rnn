from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import pkeys
from sleeprnn.common import checks

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try = 0
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    verbose = True
    grid_folder_list = None
    # -----

    dataset = load_dataset(dataset_name)  # Create dataset
    # Compute global std
    x_list = dataset.get_subset_signals(dataset.train_ids, normalize_clip=False)
    tmp_list = []
    for x in x_list:
        outlier_thr = np.percentile(np.abs(x), 99)
        tmp_signal = x[np.abs(x) <= outlier_thr]
        tmp_list.append(tmp_signal)
        this_std = tmp_signal.std()

        print(this_std)
    all_signals = np.concatenate(tmp_list)
    print('Global std:', all_signals.std())
