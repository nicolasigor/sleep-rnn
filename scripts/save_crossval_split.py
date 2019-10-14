from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import itertools
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
from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try_list = [0, 1, 2, 3]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]

    which_expert = 1
    verbose = True

    # Base parameters
    params = pkeys.default_params.copy()

    for dataset_name in dataset_name_list:
        dataset = load_dataset(dataset_name, params=params)
        test_ids = dataset.test_ids
        all_train_ids = dataset.train_ids
        val_ids_list = []
        for id_try in id_try_list:
            print('\nUsing validation split %d' % id_try)
            # Generate split
            train_ids, val_ids = utils.split_ids_list_v2(
                all_train_ids, split_id=id_try)
            val_ids_list.append(val_ids)

        splits_dict = {
            'test': test_ids,
            'train': all_train_ids,
            'val_folds': val_ids_list
        }

        pprint(splits_dict)

        filename = os.path.join(RESULTS_PATH, '%s_splits.json' % dataset_name)
        with open(filename, 'w') as outfile:
            json.dump(splits_dict, outfile)
