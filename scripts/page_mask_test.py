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
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys


if __name__ == '__main__':
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1

    params = pkeys.default_params.copy()
    dataset = load_dataset(dataset_name, params=params)

    subset_ids = dataset.train_ids[:4]
    data_feed = FeederDataset(dataset, subset_ids, task_mode, which_expert=which_expert)

    # Check page mask in parent dataset
    signal, marks = dataset.get_subject_data(
        dataset.train_ids[0],
        pages_subset=constants.N2_RECORD,
        normalization_mode=constants.N2_RECORD,
        return_page_mask=False)

