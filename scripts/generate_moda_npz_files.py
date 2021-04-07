from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint


import numpy as np
import pyedflib

project_root = os.path.abspath('..')
sys.path.append(project_root)

PATH_MODA_RAW = '/home/ntapia/Projects/Sleep_Databases/MASS_Database_2020_Full/C1'


def get_filepaths(main_path):
    files = os.listdir(main_path)
    files = [f for f in files if '.edf' in f]
    signal_files = [f for f in files if 'PSG' in f]
    states_files = [f for f in files if 'Base' in f]
    signal_files = [os.path.join(main_path, f) for f in signal_files]
    states_files = [os.path.join(main_path, f) for f in states_files]
    signal_files.sort()
    states_files.sort()
    return signal_files, states_files


def get_signal(file, chn_name):
    channel_names = file.getSignalLabels()
    channel_loc = channel_names.index(chn_name)
    check = file.getLabel(channel_loc)
    assert check == chn_name
    fs = file.samplefrequency(channel_loc)
    signal = file.readSignal(channel_loc)
    return signal, fs


if __name__ == "__main__":
    required_channel = 'C3'
    reference_channel = 'A2'  # If available, otherwise C3-LE is used as-is
    save_dir = "../resources/datasets/moda/signals_npz"
    os.makedirs(save_dir, exist_ok=True)
    print("Files will be saved at %s" % save_dir)

    # States are not necessary for the MODA dataset
    signal_files, _ = get_filepaths(PATH_MODA_RAW)
    print("%d subjects" % len(signal_files))
    for signal_f in signal_files:
        subject_id = signal_f.split("/")[-1].split(" ")[0]
        with pyedflib.EdfReader(signal_f) as file:
            channel_names = file.getSignalLabels()
            required_candidates = [chn for chn in channel_names if required_channel in chn]
            reference_candidates = [chn for chn in channel_names if reference_channel in chn]
            required_signal, required_fs = get_signal(file, required_candidates[0])
            if len(reference_candidates) > 0:
                reference_signal, reference_fs = get_signal(file, reference_candidates[0])
                assert required_fs == reference_fs
                signal = required_signal - reference_signal
                channel_extracted = '(%s)-(%s)' % (required_candidates[0], reference_candidates[0])
            else:
                signal = required_signal
                channel_extracted = required_candidates[0]
            fs = required_fs
        print('Subject %s | Channel %s at %s Hz (std %1.4f)' % (
            subject_id, channel_extracted, fs, signal.std()),
              flush=True)
        # data_dict = {
        #     'dataset_id': 'MASS-C1',
        #     'subject_id': subject_id,
        #     'sampling_rate': fs,
        #     'channel': channel_extracted,
        #     'signal': signal
        # }
        # fname = os.path.join(save_dir, "moda_%s.npz" % subject_id)
        # np.savez(fname, **data_dict)
