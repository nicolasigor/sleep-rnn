"""mass_kc.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pyedflib

from sleep.common import constants
from . import utils
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS

PATH_MASS_RELATIVE = 'mass'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'kcomplex')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

IDS_INVALID = [4, 8, 15, 16]
IDS_TEST = [2, 6, 12, 13]
# IDS_INVALID = []
# IDS_TEST = [2, 6, 12, 13, 4, 8, 15, 16]


class MassKC(Dataset):
    """This is a class to manipulate the MASS data EEG dataset.
    For K-complex events

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
        |__ 01-02-0001 KComplexesE1.edf
        |__ 01-02-0002 KComplexesE1.edf
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False):
        """Constructor"""
        # MASS parameters
        self.channel = 'EEG C3-CLE'  # Channel for SS marks
        # In MASS, we need to index by name since not all the lists are
        # sorted equally
        self.n2_id = '2'  # Character for N2 identification in hypnogram

        valid_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]
        super().__init__(
            dataset_dir=PATH_MASS_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.MASS_KC_NAME,
            all_ids=self.train_ids + self.test_ids,
            event_name=constants.KCOMPLEX,
            params=params
        )

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

            n2_pages = self._read_states(
                path_dict[KEY_FILE_STATES], signal_len)
            total_pages = int(np.ceil(signal_len / self.page_size))
            all_pages = np.arange(1, total_pages - 2, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])

            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS])
            print('Marks KC from E1: %d' % marks_1.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: marks_1
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
                '01-02-%04d KComplexesE1.edf' % subject_id)
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
        return marks

    def _read_states(self, path_states_file, signal_length):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        with pyedflib.EdfReader(path_states_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        stages_str = annotations[2]
        stages_char = [single_annot[-1] for single_annot in stages_str]
        total_annots = len(stages_char)
        # Total pages not necessarily equal to total_annots
        total_pages = int(np.ceil(signal_length / self.page_size))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int16)
        for i in range(total_annots):
            if stages_char[i] == self.n2_id:
                page_idx = int(np.round(onsets[i] / self.page_duration))
                if page_idx < total_pages:
                    n2_pages_onehot[page_idx] = 1
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
