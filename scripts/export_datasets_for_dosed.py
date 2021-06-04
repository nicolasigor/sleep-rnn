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
    fs = 256
    params = {pkeys.FS: fs}

    # MASS
    name_to_save = "mass"
    print("\nDataset: %s" % name_to_save)
    save_dir = os.path.abspath("../resources/datasets/exported_dosed/%s_files_fs_%d" % (name_to_save, fs))
    os.makedirs(save_dir, exist_ok=True)
    print("Saving directory")
    print(save_dir)
    mass_ss = load_dataset(constants.MASS_SS_NAME, params=params)
    mass_kc = load_dataset(constants.MASS_KC_NAME, params=params)
    all_ids = mass_ss.get_ids()
    for subject_id in tqdm(all_ids):
        signal = mass_ss.get_subject_signal(subject_id, which_expert=1, normalize_clip=False)
        n2_pages_from_zero = mass_ss.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
        spindle1 = mass_ss.get_subject_stamps(subject_id, which_expert=1, pages_subset=constants.N2_RECORD)
        spindle2 = mass_ss.get_subject_stamps(subject_id, which_expert=2, pages_subset=constants.N2_RECORD)
        kcomplex = mass_kc.get_subject_stamps(subject_id, which_expert=1, pages_subset=constants.N2_RECORD)
        data_to_save = {
            "signal": signal,
            "n2_pages_from_zero": n2_pages_from_zero,
            "spindle1": spindle1,
            "spindle2": spindle2,
            "kcomplex": kcomplex
        }
        fname = os.path.join(save_dir, "%s_s%s_fs_%s.npz" % (name_to_save, subject_id, fs))
        np.savez(fname, **data_to_save)
    del mass_ss
    del mass_kc

    # INTA
    name_to_save = "inta"
    print("\nDataset: %s" % name_to_save)
    save_dir = os.path.abspath("../resources/datasets/exported_dosed/%s_files_fs_%d" % (name_to_save, fs))
    os.makedirs(save_dir, exist_ok=True)
    print("Saving directory")
    print(save_dir)
    dataset = load_dataset(constants.INTA_SS_NAME, params=params)
    all_ids = dataset.get_ids()
    for subject_id in tqdm(all_ids):
        signal = dataset.get_subject_signal(subject_id, which_expert=1, normalize_clip=False)
        n2_pages_from_zero = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
        spindle = dataset.get_subject_stamps(subject_id, which_expert=1, pages_subset=constants.N2_RECORD)
        data_to_save = {
            "signal": signal,
            "n2_pages_from_zero": n2_pages_from_zero,
            "spindle": spindle,
        }
        fname = os.path.join(save_dir, "%s_s%s_fs_%s.npz" % (name_to_save, subject_id, fs))
        np.savez(fname, **data_to_save)
    del dataset

    # MODA
    name_to_save = "moda"
    print("\nDataset: %s" % name_to_save)
    save_dir = os.path.abspath("../resources/datasets/exported_dosed/%s_files_fs_%d" % (name_to_save, fs))
    os.makedirs(save_dir, exist_ok=True)
    print("Saving directory")
    print(save_dir)
    dataset = load_dataset(constants.MODA_SS_NAME, params=params)
    all_ids = dataset.get_ids()
    for subject_id in tqdm(all_ids):
        signal = dataset.get_subject_signal(subject_id, which_expert=1, normalize_clip=False)
        n2_pages_from_zero = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
        spindle = dataset.get_subject_stamps(subject_id, which_expert=1, pages_subset=constants.N2_RECORD)
        data_to_save = {
            "signal": signal,
            "n2_pages_from_zero": n2_pages_from_zero,
            "spindle": spindle,
        }
        fname = os.path.join(save_dir, "%s_s%s_fs_%s.npz" % (name_to_save, subject_id, fs))
        np.savez(fname, **data_to_save)
