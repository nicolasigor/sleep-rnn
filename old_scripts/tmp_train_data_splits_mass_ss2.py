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
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.helpers import misc
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    seed_id_list = [i for i in range(10)]

    task_mode_list = [
        constants.N2_RECORD
    ]

    dataset_name_list = [
        constants.MASS_SS_NAME
    ]
    which_expert = 1

    verbose = False

    # Base parameters
    params = pkeys.default_params.copy()
    for task_mode in task_mode_list:
        for dataset_name in dataset_name_list:
            dataset = load_dataset(dataset_name, params=params)
            ids_dict = {}
                # constants.ALL_TRAIN_SUBSET: dataset.train_ids,
                # constants.TEST_SUBSET: dataset.test_ids}
            ids_dict.update(misc.get_splits_dict(dataset, seed_id_list))
            pprint(ids_dict)
