from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import pkeys, constants


if __name__ == '__main__':
    fs = 200
    datasets_name_clip_value_list = [
        (constants.INTA_SS_NAME, 300),
        (constants.MODA_SS_NAME, 200),
        # (constants.MASS_SS_NAME, 200),
    ]
    params = {pkeys.FS: fs}

    for dataset_name, clip_value in datasets_name_clip_value_list:
        name_to_save = dataset_name.split("_")[0]
        print("\nDataset: %s" % name_to_save)
        dataset = load_dataset(dataset_name, load_checkpoint=True, params=params, verbose=False)
        all_ids = dataset.get_ids()
        all_rms = []
        for subject_id in all_ids:
            signal = dataset.get_subject_signal(subject_id, which_expert=1, normalize_clip=False)
            marks = dataset.get_subject_stamps(subject_id, which_expert=1, pages_subset=constants.N2_RECORD)
            spindles = [signal[m[0]:m[1]+1] for m in marks]
            rms = [np.sqrt(np.mean(s ** 2)) for s in spindles]
            all_rms.append(rms)
        all_rms = np.concatenate(all_rms)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        ax.hist(all_rms)
        ax.set_xlabel("Voltage ($\mu$V)")
        ax.set_title("%s (min: %1.2f, mean: %1.2f, prct5 %1.2f)" % (
            name_to_save,
            all_rms.min(),
            all_rms.mean(),
            np.percentile(all_rms, 5)
        ), fontsize=8)
        plt.show()



