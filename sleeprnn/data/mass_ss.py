"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pyedflib

from sleeprnn.common import constants
from . import utils
from . import stamp_correction
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_MARKS
from .dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM

PATH_MASS_RELATIVE = 'mass'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

IDS_INVALID = [4, 8, 15, 16]
IDS_TEST = [2, 6, 12, 13]
# IDS_INVALID = []
# IDS_TEST = [2, 6, 12, 13, 4, 8, 15, 16]


class MassSS(Dataset):
    """This is a class to manipulate the MASS data EEG dataset.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_MASS_RELATIVE
    |__ PATH_REC
        |__ 01-02-0001 PSG.edf
        |__ 01-02-0002 PSG.edf
        |__ ...
    |__ PATH_STATES
        |__ 01-02-0001 Base.edf
        |__ 01-02-0002 Base.edf
        |__ ...
    |__ PATH_MARKS
        |__ 01-02-0001 SpindleE1.edf
        |__ 01-02-0002 SpindleE1.edf
        |__ ...
        |__ 01-02-0001 SpindleE2.edf
        |__ 01-02-0002 SpindleE2.edf
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False):
        """Constructor"""
        # MASS parameters
        self.channel = 'EEG C3-CLE'  # Channel for SS marks
        # In MASS, we need to index by name since not all the lists are
        # sorted equally

        # Hypnogram parameters
        self.state_ids = np.array(['1', '2', '3', '4', 'R', 'W', '?'])
        self.unknown_id = '?'  # Character for unknown state in hypnogram
        self.n2_id = '2'  # Character for N2 identification in hypnogram

        # Sleep spindles characteristics
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        valid_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]

        print('Train size: %d. Test size: %d'
              % (len(self.train_ids), len(self.test_ids)))
        print('Train subjects: \n', self.train_ids)
        print('Test subjects: \n', self.test_ids)

        super(MassSS, self).__init__(
            dataset_dir=PATH_MASS_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.MASS_SS_NAME,
            all_ids=self.train_ids + self.test_ids,
            event_name=constants.SPINDLE,
            n_experts=2,
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
                path_dict[KEY_FILE_EEG])
            signal_len = signal.shape[0]

            n2_pages, hypnogram = self._read_states(
                path_dict[KEY_FILE_STATES], signal_len)
            total_pages = int(np.ceil(signal_len / self.page_size))
            all_pages = np.arange(1, total_pages - 2, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Hypnogram pages: %d' % hypnogram.shape[0])

            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS])
            marks_2 = self._read_marks(
                path_dict['%s_2' % KEY_FILE_MARKS])
            print('Marks SS from E1: %d, Marks SS from E2: %d'
                  % (marks_1.shape[0], marks_2.shape[0]))

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: marks_1,
                '%s_2' % KEY_MARKS: marks_2,
                KEY_HYPNOGRAM: hypnogram
            }
            data[subject_id] = ind_dict
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_file = os.path.join(
                self.dataset_dir, PATH_REC,
                '01-02-%04d PSG.edf' % subject_id)
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES,
                '01-02-%04d Base.edf' % subject_id)
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                '01-02-%04d SpindleE1.edf' % subject_id)
            path_marks_2_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                '01-02-%04d SpindleE2.edf' % subject_id)
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                '%s_1' % KEY_FILE_MARKS: path_marks_1_file,
                '%s_2' % KEY_FILE_MARKS: path_marks_2_file
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
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index(self.channel)
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            # Check
            print('Channel extracted: %s' % file.getLabel(channel_to_extract))

        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round)
        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old)
        # Now resample to the required frequency
        signal = utils.resample_signal(
            signal, fs_old=fs_old_round, fs_new=self.fs)
        signal = signal.astype(np.float32)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        with pyedflib.EdfReader(path_marks_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        marks_time = np.stack((onsets, offsets), axis=1)  # time-stamps
        # Transforms to sample-stamps
        marks = np.round(marks_time * self.fs).astype(np.int32)
        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(
            marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(
            marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _read_states(self, path_states_file, signal_length):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        # Total pages not necessarily equal to total_annots
        total_pages = int(np.ceil(signal_length / self.page_size))

        with pyedflib.EdfReader(path_states_file) as file:
            annotations = file.readAnnotations()

        onsets = np.array(annotations[0])
        durations = np.round(np.array(annotations[1]))
        stages_str = annotations[2]
        # keep only 20s durations
        valid_idx = (durations == self.page_duration)
        onsets = onsets[valid_idx]
        onsets_pages = np.round(onsets / self.page_duration).astype(np.int32)
        stages_str = stages_str[valid_idx]
        stages_char = [single_annot[-1] for single_annot in stages_str]

        # Build complete hypnogram
        total_annots = len(stages_char)

        not_unkown_ids = [
            state_id for state_id in self.state_ids
            if state_id != self.unknown_id]
        not_unkown_state_dict = {}
        for state_id in not_unkown_ids:
            state_idx = np.where(
                [stages_char[i] == state_id for i in range(total_annots)])[0]
            not_unkown_state_dict[state_id] = onsets_pages[state_idx]
        hypnogram = []
        for page in range(total_pages):
            state_not_found = True
            for state_id in not_unkown_ids:
                if page in not_unkown_state_dict[state_id] and state_not_found:
                    hypnogram.append(state_id)
                    state_not_found = False
            if state_not_found:
                hypnogram.append(self.unknown_id)
        hypnogram = np.asarray(hypnogram)

        # Extract N2 pages
        n2_pages = np.where(hypnogram == self.n2_id)[0]
        # Drop first, last and second to last page of the whole registers
        # if they where selected.
        last_page = total_pages - 1
        n2_pages = n2_pages[
            (n2_pages != 0)
            & (n2_pages != last_page)
            & (n2_pages != last_page - 1)]
        n2_pages = n2_pages.astype(np.int16)

        return n2_pages, hypnogram
