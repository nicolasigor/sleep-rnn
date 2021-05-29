from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy.io
from tqdm import tqdm

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import pkeys, constants


if __name__ == '__main__':
    fs = 100
    datasets_name_clip_value_list = [
        (constants.INTA_SS_NAME, 300),
        (constants.MODA_SS_NAME, 200),
        # (constants.MASS_SS_NAME, 200),
    ]
    params = {pkeys.FS: fs}
    format_to_export = 'mat'  # {'mat', 'npz'}

    for dataset_name, clip_value in datasets_name_clip_value_list:
        # Create directories
        name_to_save = dataset_name.split("_")[0]
        save_dir = os.path.abspath("../resources/datasets/exported/%s_%s_files" % (name_to_save, format_to_export))
        os.makedirs(save_dir, exist_ok=True)
        print("\nDataset: %s" % name_to_save)
        print("Saving directory")
        print(save_dir)
        # Load and save
        dataset = load_dataset(dataset_name, params=params)
        all_ids = dataset.get_ids()
        for subject_id in tqdm(all_ids):
            signal = dataset.get_subject_signal(subject_id, which_expert=1, normalize_clip=False)
            signal = np.clip(signal, a_min=-clip_value, a_max=clip_value)
            n2_pages_from_zero = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
            data_to_save = {"signal": signal, "n2_pages_from_zero": n2_pages_from_zero}
            fname = os.path.join(save_dir, "%s_s%s_fs_%s.%s" % (name_to_save, subject_id, fs, format_to_export))
            if format_to_export == 'mat':
                scipy.io.savemat(fname, data_to_save)
            elif format_to_export == 'npz':
                np.savez(fname, **data_to_save)
            else:
                raise ValueError("Incorrect format '%s', expected mat or npz.")
