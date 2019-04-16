"""mass_k.py: Defines the MASS class that manipulates the MASS database."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pyedflib

from . import data_ops
from . import postprocessing
from .base_dataset import BaseDataset
from .base_dataset import KEY_ID, KEY_EEG, KEY_PAGES, KEY_MARKS

PATH_MASS_RELATIVE = 'ssdata_mass'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'marks_k')
PATH_STATES = os.path.join('label', 'states')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

IDS_INVALID = [4, 8, 15, 16]
IDS_TEST = [2, 6, 12, 13]
# IDS_INVALID = []
# IDS_TEST = [2, 6, 12, 13, 4, 8, 15, 16]


class MASSK(BaseDataset):
    """This is a class to manipulate the MASS sleep EEG dataset.
    For K-complex events

    Expected directory tree inside DATA folder (see data_ops.py):

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

    def __init__(self, load_checkpoint=False):
        """Constructor"""
        # MASS parameters
        self.channel = 'EEG C3-CLE'  # Channel for SS marks
        # In MASS, we need to index by name since not all the lists are
        # sorted equally
        self.n2_char = '2'  # Character for N2 identification in hypnogram

        valid_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        test_ids = IDS_TEST
        train_ids = [i for i in valid_ids if i not in test_ids]
        super().__init__(PATH_MASS_RELATIVE, load_checkpoint, 'mass_k', 2,
                         train_ids, test_ids)

    def _load_from_files(self):
        """Loads the data from files and transforms it appropriately."""
        data_path_list = self._get_file_paths()
        data_list = []
        n_data = len(data_path_list)
        start = time.time()
        for i, path_dict in enumerate(data_path_list):
            print('\nLoading ID %d' % path_dict[KEY_ID])
            # Read data
            signal, fs_old = self._read_eeg(path_dict[KEY_FILE_EEG])
            signal_len = signal.shape[0]
            n2_pages = self._read_states(path_dict[KEY_FILE_STATES], signal_len)
            print('N2 pages: %d' % n2_pages.shape[0])
            signal = data_ops.norm_clip_eeg(signal, n2_pages, self.page_size)
            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS], fs_old)
            print('Marks (K) from E1: %d'
                  % (marks_1.shape[0]))
            # To binary sequence
            marks_1 = data_ops.inter2seq(marks_1, 0, signal_len-1)
            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_PAGES: n2_pages,
                '%s_1' % KEY_MARKS: marks_1,
                KEY_ID: path_dict[KEY_ID]
            }
            data_list.append(ind_dict)
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (ind_dict[KEY_ID], i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data_list))
        return data_list

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_path_list = []
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
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index(self.channel)
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            fs_old_round = int(np.round(fs_old))

            # Check
            print('Channel extracted: %s' % file.getLabel(channel_to_extract))
        signal = data_ops.filter_eeg(signal, fs_old)
        # We need an integer fs_old, that's why we use the rounded version. This
        # has the effect of slightly elongate the annotations of sleep spindles.
        # We provide the original fs_old so we can fix this when we read the
        # annotations. This fix is not made for the hypnogram because the effect
        # there is negligible.
        signal = data_ops.resample_eeg(signal, fs_old_round, self.fs)
        return signal, fs_old

    def _read_marks(self, path_marks_file, fs_old):
        """Loads sleep spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        with pyedflib.EdfReader(path_marks_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        marks_time = np.stack((onsets, offsets), axis=1)  # time-stamps
        # Sleep spindles are slightly longer due to the resampling using a
        # rounded fs_old.
        # Apparent sample frequency
        fs_new = fs_old * self.fs / np.round(fs_old)
        # Transforms to sample-stamps
        marks = np.round(marks_time * fs_new).astype(np.int32)
        # Combine marks that are too close according to standards
        # marks = postprocessing.combine_close_marks(
        #     marks, fs_new, self.min_ss_duration)
        # Fix durations that are outside standards
        # marks = postprocessing.filter_duration_marks(
        #     marks, fs_new, self.min_ss_duration, self.max_ss_duration)
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
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int32)
        for i in range(total_annots):
            if stages_char[i] == self.n2_char:
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
        n2_pages = n2_pages.astype(np.int32)
        return n2_pages
