from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.detection.threshold_optimization import fit_threshold
from sleeprnn.common import constants, pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    # Data settings
    dataset_name = constants.INTA_SS_NAME
    which_expert = 1
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True

    dataset = reader.load_dataset(
        dataset_name, params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=False)

    hours = 0
    n_spindles = 0
    densities = []
    durations = []
    for subject_id in dataset.all_ids:
        n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
        hours += n2_pages.size * dataset.page_duration / 3600
        marks = dataset.get_subject_stamps(subject_id, pages_subset=constants.N2_RECORD, which_expert=which_expert)
        n_spindles += marks.shape[0]
        densities.append(marks.shape[0] / (n2_pages.size * dataset.page_duration / 60))
        durations.append((marks[:, 1] - marks[:, 0] + 1) / dataset.fs)
    print("Hours: %1.4f h" % hours)
    print("N spindles: %d" % n_spindles)
    print("Mean density (spm): %1.4f" % np.mean(densities))
    print("Mean duration (s): %1.4f" % np.concatenate(durations).mean())
