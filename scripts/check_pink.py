from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.common import pkeys, constants


if __name__ == '__main__':
    dataset = reader.load_dataset(constants.PINK_NAME, load_checkpoint=True, verbose=False)

    # Check one subject
    subject_id = 1
    subject_data = dataset.read_subject_data(subject_id)
    for key in subject_data.keys():
        print(key, subject_data[key].shape, subject_data[key].dtype)
    signal, marks = dataset.get_subject_data(subject_id, normalize_clip=False)

    chosen_page = 200
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=80)
    ax.plot(signal[chosen_page, :], linewidth=0.7)
    ax.set_ylim([-150, 150])
    plt.show()
