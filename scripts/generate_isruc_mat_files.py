from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pyedflib
import scipy.io

sys.path.append('..')

from sleeprnn.helpers import reader
from sleeprnn.data import utils
from sleeprnn.common import constants


ISRUC_DATA_PATH = "../resources/datasets/unlabeled_isruc"
REC_PATH = os.path.join(ISRUC_DATA_PATH, "register")
STATE_PATH = os.path.join(ISRUC_DATA_PATH, "label/state")
save_dir = "../resources/datasets/isruc_subset_mat_files"

# Read subject IDs
invalid_subjects = ["1-025"]  # Por ahora se ignorará el '1-025'
subject_ids = [s[:5] for s in os.listdir(REC_PATH) if ".rec" in s]
subject_ids = [s for s in subject_ids if s not in invalid_subjects]
subject_ids.sort()

# Read EEG
fs = 200
pctl_to_std = 98
clip_value = 10
signal_dict = {}
for subject_id in subject_ids:
    print("")
    path_eeg_file = os.path.join(REC_PATH, "%s PSG.rec" % subject_id)
    channels_to_try = [
        ("C3-A2",),
        ("C3-M2",),
        ("C3", "A2"),
        ("C3", "M2"),
    ]
    with pyedflib.EdfReader(path_eeg_file) as file:
        channel_names = file.getSignalLabels()
        while len(channels_to_try) > 0:
            channel = channels_to_try.pop(0)
            if np.all([chn in channel_names for chn in channel]):
                break
        chn = channel_names.index(channel[0])
        signal = file.readSignal(chn)
        fs_old = file.samplefrequency(chn)
        if len(channel) == 2:
            chn2 = channel_names.index(channel[1])
            signal2 = file.readSignal(chn2)
            signal = signal - signal2
            # Check
            print('Subject %s | Channel extracted: %s minus %s at %s Hz' % (
                subject_id, file.getLabel(chn), file.getLabel(chn2), fs_old))
        else:
            # Check
            print('Subject %s | Channel extracted: %s at %s Hz' % (
                subject_id, file.getLabel(chn), fs_old))
    # fs for ISRUC will be considered 200 Hz since it is a difference only at the 15th decimal.
    fs_old_round = int(np.round(fs_old))
    # Broand bandpass filter to signal
    signal = utils.broad_filter(signal, fs_old_round)
    # Now resample to the required frequency
    if fs != fs_old_round:
        print('Resampling from %d Hz to required %d Hz' % (fs_old_round, fs))
        signal = utils.resample_signal(signal, fs_old=fs_old_round, fs_new=fs)
    else:
        print('Signal already at required %d Hz' % fs)
    signal = signal.astype(np.float32)
    # Robust normalization
    outlier_thr = np.percentile(np.abs(signal), pctl_to_std)
    tmp_signal = signal[np.abs(signal) <= outlier_thr]
    std_to_normalize = tmp_signal.std()
    signal = signal / std_to_normalize
    signal = np.clip(signal, -clip_value, clip_value)
    signal_dict[subject_id] = signal

# Read Hypnogram
n2_id = 2
original_page_duration = 30
page_duration = 20
page_size = int(fs * page_duration)
n2_pages_dict = {}
for subject_id in subject_ids:
    print("")
    signal_length = signal_dict[subject_id].size
    path_states_file = os.path.join(STATE_PATH, "%s Base E1.txt" % subject_id)
    states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
    # These are pages with 30s durations. To work with 20s pages
    # We consider the intersections with 20s divisions
    n2_pages_original = np.where(states == n2_id)[0]
    print('Original N2 pages: %d' % n2_pages_original.size)
    onsets_original = n2_pages_original * original_page_duration
    offsets_original = (n2_pages_original + 1) * original_page_duration
    total_pages = int(np.ceil(signal_length / page_size))
    n2_pages_onehot = np.zeros(total_pages, dtype=np.int16)
    for i in range(total_pages):
        onset_new_page = i * page_duration
        offset_new_page = (i + 1) * page_duration
        for j in range(n2_pages_original.size):
            intersection = (onset_new_page < offsets_original[j]) and (onsets_original[j] < offset_new_page)
            if intersection:
                n2_pages_onehot[i] = 1
                break
    n2_pages = np.where(n2_pages_onehot == 1)[0]
    # Drop first, last and second to last page of the whole registers
    # if they where selected.
    last_page = total_pages - 1
    n2_pages = n2_pages[
        (n2_pages != 0)
        & (n2_pages != last_page)
        & (n2_pages != last_page - 1)]
    n2_pages = n2_pages.astype(np.int16)
    print("Subject %s - Total N2 pages %d" % (subject_id, n2_pages.size))
    n2_pages_dict[subject_id] = n2_pages


# Adjust scale with mass global std
mass = reader.load_dataset(constants.MASS_SS_NAME)
os.makedirs(save_dir, exist_ok=True)
for subject_id in subject_ids:
    this_signal = signal_dict[subject_id]
    this_signal = this_signal * mass.global_std
    this_n2_pages_from_zero = n2_pages_dict[subject_id]
    fname = os.path.join(save_dir, "isruc_s%s_fs_%s.mat" % (subject_id, fs))
    print(this_signal.min(), this_signal.max())
    scipy.io.savemat(fname, {"signal": this_signal, "n2_pages_from_zero": this_n2_pages_from_zero})
