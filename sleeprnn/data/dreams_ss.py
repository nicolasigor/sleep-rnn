"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from sleeprnn.common import constants
from . import utils
from . import stamp_correction
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_MARKS
from .dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM

PATH_DREAMS_SS_RELATIVE = 'dreams_ss'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

IDS_TEST = [2, 4, 7]


class DreamsSS(Dataset):
    """This is a class to manipulate the DREAMS data EEG dataset.
    For sleep spindles.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_DREAMS_SS_RELATIVE
    |__ PATH_REC
        |__ excerpt1.txt
        |__ excerpt2.txt
        |__ ...
    |__ PATH_STATES
        |__ Hypnogram_excerpt1.txt
        |__ Hypnogram_excerpt2.txt
        |__ ...
    |__ PATH_MARKS
        |__ Visual_scoring1_excerpt1.txt
        |__ Visual_scoring1_excerpt2.txt
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False):
        """Constructor"""
        # DREAMS_SS parameters

        # Original sampling frequency
        self.fs_original_dict = {
            1: 100,
            2: 200,
            3: 50,
            4: 200,
            5: 200,
            6: 200,
            7: 200,
            8: 200
        }  # [Hz]

        # Hypnogram parameters
        # 5=wake
        # 4=REM stage
        # 3=sleep stage S1
        # 2=sleep stage S2
        # 1=sleep stage S3
        # 0=sleep stage S4
        # -1=sleep stage movement
        # -2 or -3 =unknow sleep stage

        self.state_ids = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5])
        self.unknown_id = -1  # Character for others (negative numbers)
        self.n2_id = 2  # Character for N2 identification in hypnogram
        self.original_state_interval = 5  # 5 [s]
        # We need to group in 20s after reading

        # Sleep spindles characteristics
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        valid_ids = [i for i in range(1, 9)]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]

        print('Train size: %d. Test size: %d'
              % (len(self.train_ids), len(self.test_ids)))
        print('Train subjects: \n', self.train_ids)
        print('Test subjects: \n', self.test_ids)

        super(DreamsSS, self).__init__(
            dataset_dir=PATH_DREAMS_SS_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.DREAMS_SS_NAME,
            all_ids=self.train_ids + self.test_ids,
            event_name=constants.SPINDLE,
            n_experts=1,
            params=params
        )

        self.global_std = self.compute_global_std(self.train_ids)
        print('Global STD:', self.global_std)

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print('\nLoading ID %d' % subject_id)
            path_dict = data_paths[subject_id]

            # Read data
            signal = self._read_eeg(
                path_dict[KEY_FILE_EEG], subject_id)

            n2_pages, hypnogram = self._read_states(path_dict[KEY_FILE_STATES])

            pages_from_hypno = hypnogram.size
            last_sample = pages_from_hypno * self.page_size
            signal = signal[:last_sample]
            signal_len = signal.shape[0]

            total_pages = int(signal_len / self.page_size)
            all_pages = np.arange(1, total_pages - 1, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Hypnogram pages: %d' % hypnogram.shape[0])

            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS])
            print('Marks SS from E1: %d' % marks_1.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: marks_1,
                KEY_HYPNOGRAM: hypnogram
            }
            data[subject_id] = ind_dict
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, i + 1, n_data, time.time() - start))
        print('%d records have been read.' % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_file = os.path.join(
                self.dataset_dir, PATH_REC,
                'excerpt%d.txt' % subject_id)
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES,
                'Hypnogram_excerpt%d.txt' % subject_id)
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                'Visual_scoring1_excerpt%d.txt' % subject_id)
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
        print(
            '%d records in %s dataset.' % (len(data_paths), self.dataset_name))
        print('Subject IDs: %s' % self.all_ids)
        return data_paths

    def _read_eeg(self, path_eeg_file, subject_id):
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        signal = np.loadtxt(path_eeg_file, skiprows=1)
        fs_old = self.fs_original_dict[subject_id]
        # Broand bandpass filter to signal
        nyquist_f = fs_old/2
        highcut = 35
        if highcut < nyquist_f:
            signal = utils.broad_filter(signal, fs_old, highcut=highcut)
        # Now resample to the required frequency
        if fs_old != self.fs:
            signal = utils.resample_signal(signal, fs_old, self.fs)
        signal = signal.astype(np.float32)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        # in seconds [start, duration]
        ss_stamps = np.loadtxt(path_marks_file, skiprows=1)
        # Transform to [start, end] in seconds
        marks_time = np.stack([ss_stamps[:, 0], ss_stamps.sum(axis=1)], axis=1)
        # Transforms to sample-stamps
        marks = np.round(marks_time * self.fs).astype(np.int32)
        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(
            marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(
            marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _read_states(self, path_states_file):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last pages of the hypnogram are ignored, since
        there is no enough context."""

        state = np.loadtxt(path_states_file, skiprows=1)
        group_by = int(self.page_duration / self.original_state_interval)
        n_pages = int(state.size / group_by)
        state_rk = np.zeros(n_pages)
        for i in range(n_pages):
            state_rk[i] = int(np.mean(state[4 * i:4 * (i + 1)]))

        # Replace every negative number with invalid id
        hypnogram = np.clip(state_rk, -1, 10).astype(np.int8)

        # Extract N2 pages
        n2_pages = np.where(hypnogram == self.n2_id)[0]
        # Drop first, and last page of the whole registers
        # if they where selected.
        last_page = n_pages - 1
        n2_pages = n2_pages[
            (n2_pages != 0)
            & (n2_pages != last_page)]
        n2_pages = n2_pages.astype(np.int16)
        return n2_pages, hypnogram
