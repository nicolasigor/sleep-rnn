from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
from pprint import pprint
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
save_dir = "../resources/datasets/cap_npz_files"

# Read subject IDs
invalid_subjects = ['5-006', '5-027', '5-031', '5-033']
subject_ids = [s[:5] for s in os.listdir(REC_PATH) if ".edf" in s]
subject_ids = [s for s in subject_ids if s not in invalid_subjects]
subject_ids.sort()
print("Found %d records\n" % len(subject_ids))

data_dict = {}

# Read EEG
fs = 200
for subject_id in subject_ids:
    data_dict[subject_id] = {'dataset_id': 'CAP Sleep Database', 'subject_id': subject_id, 'sampling_rate': fs}
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
    data_dict[subject_id]['start_time_hh_mm_ss'] = recording_start_time_hh_mm_ss.replace(".", ":")
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
        data_dict[subject_id]['channel'] = '%s minus %s' % (channel_names[channel_index], channel_names[channel_index_2])
    else:
        # Check
        print('Subject %s | Channel extracted: %s at %s Hz' % (subject_id, channel_names[channel_index], fs_old))
        data_dict[subject_id]['channel'] = channel_names[channel_index]
    signal = signal[0, :]
    if signal.std() < 0.001:
        # Signal is in volts, transform to microvolts for simplicity
        signal = 1e6 * signal
    data_dict[subject_id]['signal_physical_units'] = 'microvolts'
    # The sampling frequency is already an integer in CAP
    fs_old_round = int(np.round(fs_old))
    data_dict[subject_id]['original_sampling_rate'] = fs_old_round
    # Broand bandpass filter to signal
    signal = utils.broad_filter(signal, fs_old_round, lowcut=0.1, highcut=35)
    data_dict[subject_id]['bandpass_filter'] = 'scipy.signal.butter, 0.1-35Hz, order 3'
    # Now resample to the required frequency
    if fs != fs_old_round:
        print('Resampling from %d Hz to required %d Hz' % (fs_old_round, fs))
        signal = utils.resample_signal(signal, fs_old=fs_old_round, fs_new=fs)
        data_dict[subject_id]['resampling_function'] = 'scipy.signal.resample_poly'
    else:
        print('Signal already at required %d Hz' % fs)
        data_dict[subject_id]['resampling_function'] = 'none'
    signal = signal.astype(np.float32)
    # No robust normalization nor clipping for now
    data_dict[subject_id]['signal'] = signal

# Read hypnogram
original_page_duration = 30
for subject_id in subject_ids:
    data_dict[subject_id]['scoring_epoch_duration_seconds'] = original_page_duration
    print("")
    signal_length = data_dict[subject_id]['signal'].size
    path_states_file = os.path.join(STATE_PATH, "%s Base.txt" % subject_id)
    starting_time = data_dict[subject_id]['start_time_hh_mm_ss']
    skiprows = get_skiprows_cap_states(path_states_file)
    states_df = pd.read_csv(path_states_file, skiprows=skiprows, sep='\t')
    states_df = states_df.dropna()
    column_names = states_df.columns.values
    duration_col_name = column_names[[('Duration' in s) for s in column_names]][0]
    time_hhmmss_col_name = column_names[[('hh:mm:ss' in s) for s in column_names]][0]
    states_df['Time [s]'] = states_df[time_hhmmss_col_name].apply(lambda x: absolute_to_relative_time(x, starting_time))
    is_sleep_stage_event = ['SLEEP' in event for event in states_df['Event'].values]
    states_df = states_df.loc[is_sleep_stage_event]  # Only sleep stages
    states_df = states_df.loc[states_df[duration_col_name] == original_page_duration]  # Only 30s epochs
    stages_name_list = states_df['Sleep Stage'].values
    stages_loc_list = (states_df['Time [s]'].values / original_page_duration).astype(np.int32)  # Zero-based loc
    page_size = original_page_duration * fs
    total_pages = int(np.ceil(signal_length / page_size))
    stages = ['?'] * total_pages
    for real_loc, real_stage in zip(stages_loc_list, stages_name_list):
        try:
            stages[real_loc] = real_stage
        except IndexError:
            print("Subject %s requesting index %d when total pages is %d" % (subject_id, real_loc, total_pages))
    data_dict[subject_id]['hypnogram'] = np.asarray(stages)

# Save
os.makedirs(save_dir, exist_ok=True)
for subject_id in subject_ids:
    this_data = data_dict[subject_id]
    fname = os.path.join(save_dir, "cap_%s.npz" % subject_id)
    np.savez(fname, **this_data)
print("Files saved to")
print(save_dir)
