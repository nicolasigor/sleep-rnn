from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys

import numpy as np
import mne
import pandas as pd
import scipy.io

sys.path.append('..')

from sleeprnn.helpers import reader
from sleeprnn.data import utils
from sleeprnn.common import constants


def absolute_to_relative_time(abs_time_str, start_time_str):
    start_t = datetime.strptime(start_time_str, '%H:%M:%S')
    abs_time_str = abs_time_str.replace(".", ":")
    end_t = datetime.strptime(abs_time_str, '%H:%M:%S')
    delta = end_t - start_t
    return delta.seconds


def get_skiprows_cap_states(states_file_path):
    with open(states_file_path, 'r') as file:
        lines = file.readlines()
    skiprows = 0
    for line in lines:
        if 'Sleep Stage' in line:
            break
        skiprows += 1
    return skiprows


CAP_DATA_PATH = "../resources/datasets/unlabeled_cap"
REC_PATH = os.path.join(CAP_DATA_PATH, "register_all")
STATE_PATH = os.path.join(CAP_DATA_PATH, "label_all/state")
save_dir = "../resources/datasets/cap_mat_files"

# Read subject IDs
invalid_subjects = ['5-006', '5-027', '5-031', '5-033']
subject_ids = [s[:5] for s in os.listdir(REC_PATH) if ".edf" in s]
subject_ids = [s for s in subject_ids if s not in invalid_subjects]
subject_ids.sort()
print("Found %d records\n" % len(subject_ids))

# Read EEG
fs = 200
signal_dict = {}
starting_times_dict = {}
for subject_id in subject_ids:
    path_eeg_file = os.path.join(REC_PATH, "%s PSG.edf" % subject_id)
    channels_to_try = [
        ("C4-A1",),
        ("C4A1",),
        ("C4", "A1"),  # For 5-033
        ("C3-A2",),
        ("C3A2",),
        ("C3", "A2"),
    ]
    raw_data = mne.io.read_raw_edf(path_eeg_file, verbose=False)

    recording_start_time_hh_mm_ss = raw_data.info['meas_date'].strftime("%H:%M:%S")
    starting_times_dict[subject_id] = recording_start_time_hh_mm_ss
    fs_old = raw_data.info['sfreq']
    channel_names = raw_data.ch_names
    while len(channels_to_try) > 0:
        channel = channels_to_try.pop(0)
        if np.all([chn in channel_names for chn in channel]):
            break
    channel_index = channel_names.index(channel[0])
    signal, _ = raw_data[channel_index, :]
    if len(channel) == 2:
        channel_index_2 = channel_names.index(channel[1])
        signal2, _ = raw_data[channel_index_2, :]
        signal = signal - signal2
        # Check
        print('Subject %s | Channel extracted: %s minus %s at %s Hz' % (
        subject_id, channel_names[channel_index], channel_names[channel_index_2], fs_old))
    else:
        # Check
        print('Subject %s | Channel extracted: %s at %s Hz' % (subject_id, channel_names[channel_index], fs_old))
    signal = signal[0, :]
    if signal.std() < 0.001:
        # Signal is in volts, transform to microvolts for simplicity
        signal = 1e6 * signal
    # The sampling frequency is already an integer in CAP
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
    # No robust normalization nor clipping for now
    signal_dict[subject_id] = signal

# Read hypnogram
n2_id = 'SLEEP-S2'
original_page_duration = 30
page_duration = 20
page_size = int(fs * page_duration)
n2_pages_dict = {}
for subject_id in subject_ids:
    print("")
    signal_length = signal_dict[subject_id].size
    # print("Signal length", signal_length, "Total 30s pages", signal_length / (fs * original_page_duration))
    path_states_file = os.path.join(STATE_PATH, "%s Base.txt" % subject_id)
    starting_time = starting_times_dict[subject_id]
    skiprows = get_skiprows_cap_states(path_states_file)
    states_df = pd.read_csv(path_states_file, skiprows=skiprows, sep='\t')
    states_df = states_df.dropna()
    column_names = states_df.columns.values
    duration_col_name = column_names[[('Duration' in s) for s in column_names]][0]
    time_hhmmss_col_name = column_names[[('hh:mm:ss' in s) for s in column_names]][0]
    states_df['Time [s]'] = states_df[time_hhmmss_col_name].apply(lambda x: absolute_to_relative_time(x, starting_time))
    n2_stages = states_df.loc[states_df['Event'] == n2_id]
    n2_stages = n2_stages.loc[n2_stages[duration_col_name] == original_page_duration]
    # These are pages with 30s durations. To work with 20s pages
    # We consider the intersections with 20s divisions
    n2_pages_original = n2_stages["Time [s]"].values / original_page_duration
    # print("First page before rounding", n2_pages_original[0])
    n2_pages_original = n2_pages_original.astype(np.int32)
    # print("Max N2 page", n2_pages_original.max())
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

# Save
mass = reader.load_dataset(constants.MASS_SS_NAME)
os.makedirs(save_dir, exist_ok=True)
for subject_id in subject_ids:
    this_signal = signal_dict[subject_id]
    this_signal = np.clip(this_signal, a_min=-10*mass.global_std, a_max=10*mass.global_std)
    this_n2_pages_from_zero = n2_pages_dict[subject_id]
    fname = os.path.join(save_dir, "cap_s%s_fs_%s.mat" % (subject_id, fs))
    print(this_signal.min(), this_signal.max())
    scipy.io.savemat(fname, {"signal": this_signal, "n2_pages_from_zero": this_n2_pages_from_zero})
