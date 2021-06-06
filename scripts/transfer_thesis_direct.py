from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
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
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    # TODO: direct prediction using checkpoint from other dataset
    # TODO: For simplicity, try using both normalization: source and target std

    task_mode = constants.N2_RECORD

    configs = [
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_mass_ss"),
        dict(
            dataset_name=constants.MASS_SS_NAME, which_expert=2, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e2_n2_train_mass_ss"),
        dict(
            dataset_name=constants.MODA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_moda_ss"),
        dict(
            dataset_name=constants.INTA_SS_NAME, which_expert=1, strategy='5cv', n_seeds=3,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_inta_ss"),
    ]

    for source_config in configs:
        ckpt_folder = source_config["ckpt_folder"]
        experiment_path = os.path.join(RESULTS_PATH, 'predictions_%s' % source_config["dataset_name"], ckpt_folder)
        grid_folder_list = os.listdir(experiment_path)
        grid_folder_list.sort()

        for target_config in configs:
            if source_config["ckpt_folder"] == target_config["ckpt_folder"]:
                continue
            print("From %s to %s" % (source_config["ckpt_folder"], target_config["ckpt_folder"]))
            # Load target dataset and normalize with chosen std
            # Folds?
            for grid in grid_folder_list:
                # Load parameters
                # Load source weights
                ckpt_folder = "source ckpt folder + grid folder"



