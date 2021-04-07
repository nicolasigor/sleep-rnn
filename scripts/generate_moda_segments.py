from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

MODA_PATH = '../resources/datasets/moda'


if __name__ == "__main__":
    # Check channels extracted
    df = pd.read_csv(os.path.join(MODA_PATH, '8_MODA_primChan_180sjt.txt'), delimiter='\t')
    subject_ids = df.subject.values
    channels_from_moda = df.channel.values

    n_max = 200
    channels_from_file = []
    for subject_id, channel_from_moda in zip(subject_ids[:n_max], channels_from_moda[:n_max]):
        subject_id_short = subject_id.split(".")[0]
        data = np.load(os.path.join(MODA_PATH, 'signals_npz/moda_%s.npz' % subject_id_short))
        channel_from_file = data['channel'].item()
        channels_from_file.append(channel_from_file)
        # print(channel_from_moda, channel_from_file)
    channels_from_file = np.array(channels_from_file)
    a2_locs = np.where(channels_from_moda == 'C3-A2')[0]
    le_locs = np.where(channels_from_moda == 'C3-LE')[0]
    a2_locs_in_files = channels_from_file[a2_locs]
    le_locs_in_files = channels_from_file[le_locs]

    print("\nC2-A2 counts")
    values, counts = np.unique(a2_locs_in_files, return_counts=True)
    for v, c in zip(values, counts):
        print("Value %s with count %s" % (v, c))

    print("\nC2-LE counts")
    values, counts = np.unique(le_locs_in_files, return_counts=True)
    for v, c in zip(values, counts):
        print("Value %s with count %s" % (v, c))
