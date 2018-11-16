"""inta.py: Defines the INTA class that manipulates the INTA database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pyedflib

from . import data_ops
from . import postprocessing
from .spindle_dataset import SpindleDataset
from .spindle_dataset import KEY_ID, KEY_EEG, KEY_PAGES, KEY_MARKS

PATH_MASS_RELATIVE = 'ssdata_inta'
PATH_REC = 'register'
PATH_MARKS = 'label/marks'
PATH_STATES = 'label/states'

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

IDS_INVALID = [3]
IDS_TEST = [5, 7, 9]
NAMES = [
    'ADGU101504',
    'ALUR012904',
    'BECA011405',  # we will skip this one for now
    'BRCA062405',
    'BRLO041102',
    'BTOL083105',
    'BTOL090105',
    'CAPO092605',
    'CRCA020205',
    'ESCI031905',
    'TAGO061203']


class INTA(SpindleDataset):
    """This is a class to manipulate the INTA sleep EEG dataset.

    Expected directory tree inside DATA folder (see data_ops.py):

    PATH_MASS_RELATIVE
    |__ PATH_REC
        |__ ADGU101504.rec
        |__ ALUR012904.rec
        |__ ...
    |__ PATH_STATES
        |__ StagesOnly_ADGU101504.txt
        |__ ...
    |__ PATH_MARKS
        |__ FixedSS_ADGU101504.txt
        |__ ...
    """

    def __init__(self, load_checkpoint=False):
        """Constructor"""
        # INTA parameters
        self.channel = 1  # Channel for SS marks
        # Source format is 1:SQ4  2:SQ3  3:SQ2  4:SQ1  5:REM  6:WA
        self.n2_char = 3  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        valid_ids = [i for i in range(1, 12) if i not in IDS_INVALID]
        test_ids = IDS_TEST
        train_ids = [i for i in valid_ids if i not in test_ids]
        super().__init__(PATH_MASS_RELATIVE, load_checkpoint, 'inta', 1,
                         train_ids, test_ids)

    def _load_from_files(self):
        """Loads the data from files and transforms it appropriately."""
        data_path_list = self._get_file_paths()
        data_list = []
        n_data = len(data_path_list)
        start = time.time()
        for i, path_dict in enumerate(data_path_list):
            subject_id = path_dict[KEY_ID]
            print('\nLoading ID %d, name %s'
                  % (subject_id, NAMES[subject_id - 1]))
            # Read data
            signal = self._read_eeg(path_dict[KEY_FILE_EEG])
            signal_len = signal.shape[0]
            n2_pages = self._read_states(path_dict[KEY_FILE_STATES], signal_len)
            print('N2 pages: %d' % n2_pages.shape[0])
            signal = data_ops.norm_clip_eeg(signal, n2_pages, self.page_size)
            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS])
            print('Marks from E1: %d' % marks_1.shape[0])
            # To binary sequence
            marks_1 = data_ops.inter2seq(marks_1, 0, signal_len-1)
            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_PAGES: n2_pages,
                '%s_1' % KEY_MARKS: marks_1,
                KEY_ID: subject_id
            }
            data_list.append(ind_dict)
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (ind_dict[KEY_ID], i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data_list))
        return data_list

    def _get_file_paths(self):
        """Returns a list of dicts containing paths for loading the database."""
        # Build list of paths
        data_path_list = []
        for subject_id in self.all_ids:
            subject_name = NAMES[subject_id - 1]
            path_eeg_file = os.path.join(
                self.dataset_dir, PATH_REC,
                '%s.rec' % subject_name)
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES,
                'StagesOnly_%s.txt' % subject_name)
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                'FixedSS_%s.txt' % subject_name)
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                '%s_1' % KEY_FILE_MARKS: path_marks_1_file,
                KEY_ID: subject_id
            }
            # Check paths
            for key in ind_dict:
                if key != KEY_ID:
                    if not os.path.isfile(ind_dict[key]):
                        raise FileNotFoundError(
                            'File not found: %s' % ind_dict[key])
            data_path_list.append(ind_dict)
        print('%d records in %s dataset.' % (len(data_path_list), self.name))
        print('Subject IDs: %s' % self.all_ids)
        return data_path_list

    def _read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file', does filtering."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            signal = file.readSignal(self.channel)
        signal = data_ops.filter_eeg(signal, self.fs)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads sleep spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        # Recovery sample-stamps
        marks_file = np.loadtxt(path_marks_file, dtype='i', delimiter=' ')
        marks = marks_file[marks_file[:, 5] == self.channel][:, [0, 1]]
        marks = np.round(marks).astype(np.int32)
        # Combine marks that are too close according to standards
        marks = postprocessing.combine_close_marks(
            marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = postprocessing.filter_duration_marks(
            marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _read_states(self, path_states_file, signal_length):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
        # These are pages with 30s durations. To work with 20s pages
        # We consider the intersections with 20s divisions
        n2_pages_original = np.where(states == self.n2_char)[0]
        print('Original N2 pages: %d' % n2_pages_original.size)
        onsets_original = n2_pages_original * self.original_page_duration
        offsets_original = (n2_pages_original+1) * self.original_page_duration
        total_pages = int(np.ceil(signal_length / self.page_size))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int32)
        for i in range(total_pages):
            onset_new_page = i * self.page_duration
            offset_new_page = (i+1) * self.page_duration
            for j in range(n2_pages_original.size):
                intersection = (onset_new_page < offsets_original[j]) \
                               and (onsets_original[j] < offset_new_page)
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
        n2_pages = n2_pages.astype(np.int32)
        return n2_pages
