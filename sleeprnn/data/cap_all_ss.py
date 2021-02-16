"""isruc_ss.py: Defines the CAP class that manipulates the CAP database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time

import mne
import numpy as np
import pandas as pd

from sleeprnn.common import constants
from . import utils
from . import stamp_correction
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS

PATH_CAP_RELATIVE = 'unlabeled_cap'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

# normal subjects, 23-42 year old
IDS_VALID = [
    '1-001',
    '1-002',
    '1-003',
    '1-004',
    '1-005',
    '1-006',
    '1-007',
    '1-008',
    '1-009',
    '1-010',
    '1-011',
    '1-012',
    '1-013',
    '1-014',
    '1-015',
    '1-016',
    '2-002',
    '4-001',
    '4-003',
    '4-005',
    '5-003',
    '5-004',
    '5-005',
    '5-007',
    '5-009',
    '5-010',
    '5-011',
    '5-015',
    '5-016',
    '5-017',
    '5-018',
    '5-019',
    '5-020',
    '5-021',
    '5-023',
    '5-028',
    '5-030',
    '5-034',
    '5-036',
    '5-038',
    '5-039'
]


class CapAllSS(Dataset):
    """This is a class to manipulate the CAP data EEG dataset.

    The sleep spindle marks are detections made by the A7 algorithm:
    Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., & Warby, S. C. (2019).
    A sleep spindle detection algorithm that emulates human expert spindle scoring.
    Journal of neuroscience methods, 316, 3-11.
    The four parameters were fitted on MASS-SS2.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_CAP_RELATIVE
    |__ PATH_REC
        |__ 1-003 PSG.edf
        |__ 1-004 PSG.edf
        |__ ...
    |__ PATH_STATES
        |__ 1-003 Base.txt
        |__ ...
    |__ PATH_MARKS
        |__ EventDetection_s1-001_*.txt
        |__ ...
    """

    def __init__(
            self, params=None, load_checkpoint=False, verbose=True, external_global_std=None):
        """Constructor"""
        # CAP parameters
        self.channels_to_try = [
            "C4-A1",
            "C4A1",
            "C3-A2",
            "C3A2",
        ]  # Channel for SS
        self.n2_id = 'SLEEP-S2'  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        # Sleep spindles characteristics
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        self.train_ids = IDS_VALID
        self.train_ids.sort()

        if verbose:
            print('Train size: %d.' % len(self.train_ids))
            print('Train subjects: \n', self.train_ids)

        super(CapAllSS, self).__init__(
            dataset_dir=PATH_CAP_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.CAP_ALL_SS_NAME,
            all_ids=self.train_ids,
            event_name=constants.SPINDLE,
            params=params,
            verbose=verbose
        )
        if external_global_std is not None:
            self.global_std = external_global_std
            if verbose:
                print('Global STD set externally:', self.global_std)
        else:
            self.global_std = self.compute_global_std(self.train_ids)
            if verbose:
                print('Global STD computed:', self.global_std)

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print('\nLoading ID %s' % subject_id)
            path_dict = data_paths[subject_id]

            # Read data
            signal, recording_start_time = self._read_eeg(path_dict[KEY_FILE_EEG])
            signal_len = signal.shape[0]

            n2_pages = self._read_states(path_dict[KEY_FILE_STATES], signal_len, recording_start_time)
            total_pages = int(np.ceil(signal_len / self.page_size))
            all_pages = np.arange(1, total_pages - 2, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])

            marks_1 = self._read_marks(path_dict['%s_1' % KEY_FILE_MARKS])
            print('Marks SS from A7: %d' % marks_1.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: marks_1
            }
            data[subject_id] = ind_dict
            print('Loaded ID %s (%02d/%02d ready). Time elapsed: %1.4f [s]' % (
                subject_id, i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_file = os.path.join(self.dataset_dir, PATH_REC, "%s PSG.edf" % subject_id)
            path_states_file = os.path.join(self.dataset_dir, PATH_STATES, "%s Base.txt" % subject_id)
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                'EventDetection_s%s_absSigPow(1.75)_relSigPow(1.6)_sigCov(1.8)_sigCorr(0.75).txt' % subject_id)
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                '%s_1' % KEY_FILE_MARKS: path_marks_1_file
            }
            # Check paths
            for key in ind_dict:
                if not os.path.isfile(ind_dict[key]):
                    print(
                        'File not found: %s' % ind_dict[key])
            data_paths[subject_id] = ind_dict
        print('%d records in %s dataset.' % (len(data_paths), self.dataset_name))
        print('Subject IDs: %s' % self.all_ids)
        return data_paths

    def _read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file', does filtering."""
        raw_data = mne.io.read_raw_edf(path_eeg_file, verbose=False)
        recording_start_time_hh_mm_ss = raw_data.info['meas_date'].strftime("%H:%M:%S")
        channel_names = raw_data.ch_names
        channels_to_try = self.channels_to_try.copy()
        while len(channels_to_try) > 0:
            channel = channels_to_try.pop(0)
            if channel in channel_names:
                break
        channel_index = channel_names.index(channel)
        signal, _ = raw_data[channel_index, :]
        signal = signal[0, :]
        if signal.std() < 0.01:
            # Signal is in volts, transform to microvolts for simplicity
            signal = 1e6 * signal
        fs_old = raw_data.info['sfreq']
        # Check
        print('Channel extracted: %s' % raw_data.ch_names[channel_index])
        # The sampling frequency is already an integer in CAP
        fs_old = int(np.round(fs_old))
        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old)
        # Now resample to the required frequency
        if self.fs != fs_old:
            print('Resampling from %d Hz to required %d Hz' % (fs_old, self.fs))
            signal = utils.resample_signal(signal, fs_old=fs_old, fs_new=self.fs)
        else:
            print('Signal already at required %d Hz' % self.fs)
        signal = signal.astype(np.float32)
        return signal, recording_start_time_hh_mm_ss

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        pred_data = pd.read_csv(path_marks_file, sep='\t')
        # We substract 1 to translate from matlab to numpy indexing system
        start_samples = pred_data.start_sample.values - 1
        end_samples = pred_data.end_sample.values - 1
        marks = np.stack([start_samples, end_samples], axis=1).astype(np.int32)

        # Sample-stamps assume 200Hz sampling rate
        if self.fs != 200:
            print('Correcting marks from 200 Hz to %d Hz' % self.fs)
            # We need to transform the marks to the new sampling rate
            marks_time = marks.astype(np.float32) / 200.0
            # Transform to sample-stamps
            marks = np.round(marks_time * self.fs).astype(np.int32)

        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _read_states(self, path_states_file, signal_length, recording_start_time):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        skiprows = self._get_skiprows_cap_states(path_states_file)
        states_df = pd.read_csv(path_states_file, skiprows=skiprows, sep='\t')
        states_df = states_df.dropna()
        column_names = states_df.columns.values
        duration_col_name = column_names[[('Duration' in s) for s in column_names]][0]
        time_hhmmss_col_name = column_names[[('hh:mm:ss' in s) for s in column_names]][0]
        states_df['Time [s]'] = states_df[time_hhmmss_col_name].apply(
            lambda x: self._absolute_to_relative_time(x, recording_start_time))
        n2_stages = states_df.loc[states_df['Event'] == self.n2_id]
        n2_stages = n2_stages.loc[n2_stages[duration_col_name] == self.original_page_duration]
        # These are pages with 30s durations. To work with 20s pages
        # We consider the intersections with 20s divisions
        n2_pages_original = n2_stages["Time [s]"].values / self.original_page_duration
        n2_pages_original = n2_pages_original.astype(np.int32)
        print('Original N2 pages: %d' % n2_pages_original.size)
        onsets_original = n2_pages_original * self.original_page_duration
        offsets_original = (n2_pages_original + 1) * self.original_page_duration
        total_pages = int(np.ceil(signal_length / self.page_size))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int16)
        for i in range(total_pages):
            onset_new_page = i * self.page_duration
            offset_new_page = (i + 1) * self.page_duration
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
        return n2_pages

    def _absolute_to_relative_time(self, abs_time_str, start_time_str):
        start_t = datetime.strptime(start_time_str, '%H:%M:%S')
        end_t = datetime.strptime(abs_time_str, '%H:%M:%S')
        delta = end_t - start_t
        return delta.seconds

    def _get_skiprows_cap_states(self, states_file_path):
        with open(states_file_path, 'r') as file:
            lines = file.readlines()
        skiprows = 0
        for line in lines:
            if 'Sleep Stage' in line:
                break
            skiprows += 1
        return skiprows
