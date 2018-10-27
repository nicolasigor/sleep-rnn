"""mass.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
import pyedflib

from .data_ops import inter2seq
from .data_ops import filter_eeg, resample_eeg, norm_clip_eeg
from .data_ops import PATH_DATA
from utils.constants import TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET

PATH_MASS = os.path.join(PATH_DATA, 'ssdata_mass')
PATH_REC = 'register'
PATH_MARKS_1 = 'label/marks/e1'
PATH_MARKS_2 = 'label/marks/e2'
PATH_STATES = 'label/states'
PATH_CHECKPOINT = 'checkpoint'

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'
KEY_REG_ID = 'reg_id'
KEY_EEG = 'signal'
KEY_PAGES = 'pages'
KEY_MARKS = 'marks'

IDS_INVALID = [4, 8, 15, 16, 18]
IDS_TRAIN = [1, 5, 7, 9, 10, 14, 17, 19]
IDS_VAL = [3, 11]
IDS_TEST = [2, 6, 12, 13]


# TODO: Check subject 18 (performance issue)
class MASS(object):

    def __init__(self, load_checkpoint=False):
        self.name = 'MASS'
        self.fs = 200  # Sampling frequency [Hz] to be used (not the original one)
        self.page_duration = 20  # Time of window page [s]
        self.page_size = int(self.page_duration * self.fs)
        self.min_ss_duration = 0.3  # Minimum feasible duration of SS in seconds
        self.max_ss_duration = 3  # Maximum feasible duration of SS in seconds
        self.channel = 13  # Channel for SS marks, C3 in 13, F3 in 22
        self.n2_char = '2'  # Character for N2 identification in hypnogram
        self.data = self.load_data(load_checkpoint)
        n_pages = np.sum([ind[KEY_PAGES].shape[0] for ind in self.data])
        print("\nPages in %s dataset: %d" % (self.name, n_pages))

    def load_data(self, load_checkpoint):
        """Loads data either from a checkpoint or from scratch."""
        if load_checkpoint and self.exists_checkpoint():
            print('Loading from checkpoint')
            data = self.load_from_checkpoint()
        else:
            if load_checkpoint:
                print('A checkpoint does not exist. Loading from files instead.')
            else:
                print('Loading from files.')
            data = self.load_from_files()
        return data

    def get_file_paths(self):
        """Returns a list of dictionaries containing relevant paths for loading the database."""
        # Only those registers that have marks from two experts
        reg_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        # Build list of paths
        data_path_list = []
        for i, reg_id in enumerate(reg_ids):
            path_eeg_file = os.path.join(PATH_MASS, PATH_REC, '01-02-%04d PSG.edf' % reg_id)
            path_states_file = os.path.join(PATH_MASS, PATH_STATES, '01-02-%04d Base.edf' % reg_id)
            path_marks_1_file = os.path.join(PATH_MASS, PATH_MARKS_1, '01-02-%04d SpindleE1.edf' % reg_id)
            path_marks_2_file = os.path.join(PATH_MASS, PATH_MARKS_2, '01-02-%04d SpindleE2.edf' % reg_id)
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                '%s_1' % KEY_FILE_MARKS: path_marks_1_file,
                '%s_2' % KEY_FILE_MARKS: path_marks_2_file,
                KEY_REG_ID: reg_id
            }
            data_path_list.append(ind_dict)
        print('%d records in %s dataset.' % (len(data_path_list), self.name))
        print('Subject IDs: %s' % reg_ids)
        return data_path_list

    def save_checkpoint(self):
        """Saves a pickle file containing the loaded data."""
        filename = os.path.join(PATH_MASS, PATH_CHECKPOINT, '%s.pickle' % self.name)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def exists_checkpoint(self):
        """Checks whether the pickle file with the checkpoint exists."""
        filename = os.path.join(PATH_MASS, PATH_CHECKPOINT, '%s.pickle' % self.name)
        return os.path.isfile(filename)

    def load_from_checkpoint(self):
        """Loads the pickle file containing the loaded data."""
        filename = os.path.join(PATH_MASS, PATH_CHECKPOINT, '%s.pickle' % self.name)
        with open(filename, 'rb') as handle:
            data_list = pickle.load(handle)
        return data_list

    def load_from_files(self):
        """Loads the data from files and transforms it appropriately."""
        data_path_list = self.get_file_paths()
        data_list = []
        n_data = len(data_path_list)
        start = time.time()
        for i, path_dict in enumerate(data_path_list):
            print('\nLoading ID %d' % path_dict[KEY_REG_ID])
            # Read data
            signal, fs_old = self.read_eeg(path_dict[KEY_FILE_EEG])
            signal_length = signal.shape[0]
            n2_pages = self.read_states(path_dict[KEY_FILE_STATES], signal_length)
            print('N2 pages: %d' % n2_pages.shape[0])
            signal = norm_clip_eeg(signal, n2_pages, self.page_size)
            marks_1 = self.read_marks(path_dict['%s_1' % KEY_FILE_MARKS], fs_old)
            marks_2 = self.read_marks(path_dict['%s_2' % KEY_FILE_MARKS], fs_old)
            print('Marks from E1: %d, Marks from E2: %d' % (marks_1.shape[0], marks_2.shape[0]))
            marks_1 = inter2seq(marks_1, 0, signal_length-1)  # To binary sequence
            marks_2 = inter2seq(marks_2, 0, signal_length-1)  # To binary sequence
            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_PAGES: n2_pages,
                '%s_1' % KEY_MARKS: marks_1,
                '%s_2' % KEY_MARKS: marks_2,
                KEY_REG_ID: path_dict[KEY_REG_ID]
            }
            data_list.append(ind_dict)
            print('Loaded ID %d (%d/%d ready). Time elapsed: %1.4f [s]'
                  % (ind_dict[KEY_REG_ID], i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data_list))
        return data_list

    def read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file'. The signal is filtered and resampled."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            signal = file.readSignal(self.channel)
            fs_old = file.samplefrequency(self.channel)
            fs_old_round = int(np.round(fs_old))
        signal = filter_eeg(signal, self.fs)
        # We need an integer fs_old, that's why we use the rounded version. This has the effect of
        # slightly elongate the annotations of sleep spindles. We provide the original fs_old
        # so we can fix this when we read the annotations. This fix is not made for the hypnogram because
        # the effect there is negligible.
        signal = resample_eeg(signal, fs_old_round, self.fs)
        return signal, fs_old

    def read_marks(self, path_marks_file, fs_old):
        """Loads sleep spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed. Returns the sample-stamps of each mark."""
        with pyedflib.EdfReader(path_marks_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        marks_time = np.stack((onsets, offsets), axis=1)  # time-stamps
        # Sleep spindles are slightly longer due to the resampling using a rounded fs_old.
        fs_new = fs_old * self.fs / np.round(fs_old)  # Apparent sample frequency
        marks = np.round(marks_time * fs_new).astype(np.int32)  # Transforms to sample-stamps
        # Remove too short and too long spindles
        feasible_idx = np.where((durations >= self.min_ss_duration) & (durations <= self.max_ss_duration))[0]
        marks = marks[feasible_idx, :]
        return marks

    def read_states(self, path_states_file, signal_length):
        """Loads hypnogram from 'path_states_file'. Only n2 pages indices are returned.
        First, last and second to last pages of the hypnogram are ignored, since there is no enough context."""
        with pyedflib.EdfReader(path_states_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        stages_str = annotations[2]
        stages_char = [single_annot[-1] for single_annot in stages_str]
        total_annots = len(stages_char)
        total_pages = int(np.ceil(signal_length / self.page_size))  # Not necessarily equal to total_annots
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int32)
        for i in range(total_annots):
            if stages_char[i] == self.n2_char:
                page_idx = int(np.round(onsets[i] / self.page_duration))
                if page_idx < total_pages:
                    n2_pages_onehot[page_idx] = 1
        n2_pages = np.where(n2_pages_onehot == 1)[0]
        # Drop first, last and second to last page of the whole registers if they where selected
        last_page = total_pages - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page) & (n2_pages != last_page - 1)]
        n2_pages = n2_pages.astype(np.int32)
        return n2_pages

    # TODO: method to obtain the n2 segments of a list of subjects.
    def get_subject_data(self, id, augmented_page=False, border_size=0, which_mark=1):
        pass

    # def preprocessing_eeg(self, signal, n2_pages):
    #     # Clip at -250, 250
    #     thr = 250
    #     signal = np.clip(signal, -thr, thr)
    #     # Concatenate every n2 page
    #     n2_list = []
    #     for page in n2_pages:
    #         sample_start = page * self.page_size
    #         sample_end = (page + 1) * self.page_size
    #         n2_signal = signal[sample_start:sample_end]
    #         n2_list.append(n2_signal)
    #     n2_signal = np.concatenate(n2_list, axis=0)
    #     # Compute mean and std for n2 pages
    #     data_mean = np.mean(n2_signal)
    #     data_std = np.std(n2_signal)
    #     # Normalization
    #     new_signal = (signal - data_mean) / data_std
    #     return new_signal
    #
    # def get_subset(self, subset_name):
    #     # Select subset
    #     if subset_name == VAL_SUBSET:
    #         data_list = self.data_val
    #     elif subset_name == TEST_SUBSET:
    #         data_list = self.data_test
    #     elif subset_name == TRAIN_SUBSET:
    #         data_list = self.data_train
    #     else:
    #         raise Exception("Invalid subset_name '%s'. Expected '%s', '%s', or '%s'." %
    #                         (subset_name, TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET))
    #     return data_list
    #
    # def get_augmented_numpy_subset(self, subset_name, mark_mode, border_sec):
    #     # Get augmented pages for random cropping.
    #     # border_sec: Seconds to be added at both borders of the augmented page.
    #     data_list = self.get_subset(subset_name)
    #     border_size = border_sec * self.fs
    #     features = []
    #     labels = []
    #     # Iterate over registers
    #     for i in range(len(data_list)):
    #         ind_dict = data_list[i]
    #         signal = ind_dict['signal']
    #         n2_pages = ind_dict['pages']
    #         if mark_mode == 1:
    #             marks = ind_dict['marks_1']
    #         elif mark_mode == 2:
    #             marks = ind_dict['marks_2']
    #         else:
    #             raise Exception("Invalid mark_mode %s. Expected 1 or 2." % mark_mode)
    #         # Iterate over pages
    #         for page in n2_pages:
    #             offset = page * self.page_size
    #             start_sample = int(offset - self.page_size/2 - border_size)
    #             end_sample = int(start_sample + 2*self.page_size + 2*border_size)
    #             augmented_page_signal = signal[start_sample:end_sample]
    #             augmented_page_labels = marks[start_sample:end_sample]
    #             features.append(augmented_page_signal)
    #             labels.append(augmented_page_labels)
    #     features_np = np.stack(features, axis=0)
    #     labels_np = np.stack(labels, axis=0)
    #     return features_np, labels_np
