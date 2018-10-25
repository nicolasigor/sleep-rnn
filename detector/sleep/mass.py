from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import time

import numpy as np
import pyedflib
import scipy.signal as sp_signal

from .data_ops import seq2inter, inter2seq
from neuralnet.constants import TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET

PATH_THIS_PACKAGE = os.path.dirname(os.path.abspath(__file__))
PATH_DATABASE = os.path.abspath(os.path.join(PATH_THIS_PACKAGE, "../../data/ssdata_mass"))
PATH_REC = "register"
PATH_MARKS_1 = "label/marks/e1"
PATH_MARKS_2 = "label/marks/e2"
PATH_STATES = "label/states"


class MASS(object):

    def __init__(self, load_checkpoint=False):
        print('Loading database from %s' % PATH_DATABASE)
        self.name = "MASS"
        self.fs = 200  # Sampling frequency [Hz] to be used (not the original one)
        self.dur_page = 20  # Time of window page [s]
        self.page_size = int(self.dur_page * self.fs)
        self.min_ss_dur = 0.3  # Minimum feasible duration of SS
        self.max_ss_dur = 3  # Maximum feasible duration of SS
        self.channel = 13  # Channel for SS marks, # C3 in 13, F3 in 22
        self.n2_char = '2'  # Character for N2 identification in hipnogram

        if load_checkpoint:
            print("\nLoading %s from checkpoint" % self.name)
            self.data_train = self.load_subset_checkpoint(TRAIN_SUBSET)
            self.data_val = self.load_subset_checkpoint(VAL_SUBSET)
            self.data_test = self.load_subset_checkpoint(TEST_SUBSET)
        else:
            # Get train, val and test files
            data_path_list = self.get_file_paths()
            train_path_list, val_path_list, test_path_list = self.random_split(data_path_list)
            print("\nLoading train set...")
            self.data_train = self.load_data(train_path_list)
            print("\nLoading val set...")
            self.data_val = self.load_data(val_path_list)
            print("\nLoading test set...")
            self.data_test = self.load_data(test_path_list)

        n_pages_train = np.sum([ind['pages'].shape[0] for ind in self.data_train])
        print("\nPages in train set: %d" % n_pages_train)
        n_pages_val = np.sum([ind['pages'].shape[0] for ind in self.data_val])
        print("Pages in val set: %d" % n_pages_val)
        n_pages_test = np.sum([ind['pages'].shape[0] for ind in self.data_test])
        print("Pages in test set: %d" % n_pages_test)
        total_pages = n_pages_train + n_pages_val + n_pages_test
        print("\nPages in %s dataset: %d" % (self.name, total_pages))

    def get_fs(self):
        return self.fs

    def get_page_size(self):
        return self.page_size

    def get_page_duration(self):
        return self.dur_page

    def get_file_paths(self):
        # Only those registers that have marks from two experts
        reg_ids = [i for i in range(1, 20) if i not in [4, 8, 15, 16]]
        # Build list of paths
        data_path_list = []
        for i in range(len(reg_ids)):
            path_eeg_file = os.path.join(PATH_DATABASE, PATH_REC, '01-02-%04d PSG.edf' % reg_ids[i])
            path_states_file = os.path.join(PATH_DATABASE, PATH_STATES, '01-02-%04d Base.edf' % reg_ids[i])
            path_marks_1_file = os.path.join(PATH_DATABASE, PATH_MARKS_1, '01-02-%04d SpindleE1.edf' % reg_ids[i])
            path_marks_2_file = os.path.join(PATH_DATABASE, PATH_MARKS_2, '01-02-%04d SpindleE2.edf' % reg_ids[i])
            # Save data
            ind_dict = {'file_eeg': path_eeg_file, 'file_states': path_states_file,
                        'file_marks_1': path_marks_1_file, 'file_marks_2': path_marks_2_file,
                        'reg_id': reg_ids[i]}
            data_path_list.append(ind_dict)
        print('%d records in %s dataset.' % (len(data_path_list), self.name))
        return data_path_list

    def random_split(self, data_path_list):
        # random_perm = [10,  1,  4,  9,  2,  8, 13,  5, 11,  6,  3, 14, 12,  0,  7]
        # test_idx = random_perm[0:4]
        # val_idx = random_perm[4:7]
        # train_idx = random_perm[7:]
        test_idx = [1,  4,  9, 10]
        # val_idx = [2,  8, 13]
        val_idx = [2, 8]
        train_idx = [5, 11,  6,  3, 14, 12,  0,  7]

        train_path_list = [data_path_list[i] for i in train_idx]
        val_path_list = [data_path_list[i] for i in val_idx]
        test_path_list = [data_path_list[i] for i in test_idx]
        train_reg_ids = [data_path_list[i]['reg_id'] for i in train_idx]
        val_reg_ids = [data_path_list[i]['reg_id'] for i in val_idx]
        test_reg_ids = [data_path_list[i]['reg_id'] for i in test_idx]
        print('Train set size: %d -- Records    ID: %s' % (len(train_path_list), train_reg_ids))
        print('Val set size: %d -- Records    ID: %s' % (len(val_path_list), val_reg_ids))
        print('Test set size: %d -- Records    ID: %s' % (len(test_path_list), test_reg_ids))
        return train_path_list, val_path_list, test_path_list

    def save_checkpoint(self):
        self.save_subset_checkpoint(self.data_train, TRAIN_SUBSET)
        self.save_subset_checkpoint(self.data_val, VAL_SUBSET)
        self.save_subset_checkpoint(self.data_test, TEST_SUBSET)

    def save_subset_checkpoint(self, data_list, subset_name):
        filename = os.path.join(PATH_DATABASE, "checkpoint_%s/%s.pickle" % (self.name, subset_name))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_subset_checkpoint(self, subset_name):
        filename = os.path.join(PATH_DATABASE, "checkpoint_%s/%s.pickle" % (self.name, subset_name))
        with open(filename, 'rb') as handle:
            data_list = pickle.load(handle)
        return data_list

    def load_data(self, data_path_list):
        data_list = []
        n_data = len(data_path_list)
        start = time.time()
        for i in range(n_data):
            # Read EEG Signal, States, and Marks
            path_eeg_file = data_path_list[i]['file_eeg']
            path_states_file = data_path_list[i]['file_states']
            path_marks_1_file = data_path_list[i]['file_marks_1']
            path_marks_2_file = data_path_list[i]['file_marks_2']

            signal, fs_old = self.read_eeg(path_eeg_file)
            n2_pages = self.read_states(path_states_file, signal.shape[0])
            marks_1 = self.read_marks(path_marks_1_file, fs_old)
            marks_2 = self.read_marks(path_marks_2_file, fs_old)

            # Clip-Normalize eeg signal
            signal = self.preprocessing_eeg(signal, n2_pages)
            # Transform marks into 0_1 format
            marks_1 = inter2seq(marks_1, 0, signal.shape[0] - 1)
            marks_2 = inter2seq(marks_2, 0, signal.shape[0] - 1)
            # Save data
            reg_id = data_path_list[i]['reg_id']
            ind_dict = {'signal': signal, 'pages': n2_pages,
                        'marks_1': marks_1, 'marks_2': marks_2,
                        'reg_id': reg_id}
            data_list.append(ind_dict)
            print('%d/%d ready, time elapsed: %1.4f [s]' % (i+1, n_data, time.time() - start))
        print('%d records have been read.' % len(data_list))
        return data_list

    def read_eeg(self, path_eeg_file):
        file = pyedflib.EdfReader(path_eeg_file)
        signal = file.readSignal(self.channel)
        fs_old = file.samplefrequency(self.channel)
        fs_old_round = int(np.round(fs_old))
        file._close()
        del file

        # Resample
        gcd_freqs = math.gcd(self.fs, fs_old_round)
        up = int(self.fs / gcd_freqs)
        down = int(fs_old_round / gcd_freqs)
        signal = sp_signal.resample_poly(signal, up, down)
        signal = np.array(signal, dtype=np.float32)
        return signal, fs_old

    def read_marks(self, path_marks_file, fs_old):
        file = pyedflib.EdfReader(path_marks_file)
        annotations = file.readAnnotations()
        file._close()
        del file
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        # Remove too short and too long marks
        feasible_idx = np.where((durations >= self.min_ss_dur) & (durations <= self.max_ss_dur))
        onsets = onsets[feasible_idx]
        durations = durations[feasible_idx]
        offsets = onsets + durations
        print("Marks: %d" % onsets.shape[0])
        # Translate to a sample step:
        fs_new = fs_old * self.fs / np.round(fs_old)
        start_samples = np.array(np.round(onsets * fs_new), dtype=np.int32)
        end_samples = np.array(np.round(offsets * fs_new), dtype=np.int32)
        marks = np.stack((start_samples, end_samples), axis=1)
        return marks

    def read_states(self, path_states_file, signal_length):
        file = pyedflib.EdfReader(path_states_file)
        annotations = file.readAnnotations()
        file._close()
        del file
        onsets = np.array(annotations[0])
        stages_str = annotations[2]
        stages_char = [single_annot[-1] for single_annot in stages_str]
        total_annots = len(stages_char)
        total_pages = int(np.ceil(signal_length / self.page_size))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int32)
        for i in range(total_annots):
            if stages_char[i] == self.n2_char:
                page_idx = int(np.round(onsets[i] / self.dur_page))
                n2_pages_onehot[page_idx] = 1
        n2_pages = np.where(n2_pages_onehot == 1)[0]
        # Drop first, last and second to last page of the whole registers if they where selected
        last_page = total_pages - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page) & (n2_pages != last_page - 1)]
        n2_pages = np.array(n2_pages, dtype=np.int32)
        return n2_pages

    # TODO: filter eeg in [0.5, 35] range
    def preprocessing_eeg(self, signal, n2_pages):
        # Clip at -250, 250
        thr = 250
        signal = np.clip(signal, -thr, thr)
        # Concatenate every n2 page
        n2_list = []
        for page in n2_pages:
            sample_start = page * self.page_size
            sample_end = (page + 1) * self.page_size
            n2_signal = signal[sample_start:sample_end]
            n2_list.append(n2_signal)
        n2_signal = np.concatenate(n2_list, axis=0)
        # Compute mean and std for n2 pages
        data_mean = np.mean(n2_signal)
        data_std = np.std(n2_signal)
        # Normalization
        new_signal = (signal - data_mean) / data_std
        return new_signal

    def get_subset(self, subset_name):
        # Select subset
        if subset_name == VAL_SUBSET:
            data_list = self.data_val
        elif subset_name == TEST_SUBSET:
            data_list = self.data_test
        elif subset_name == TRAIN_SUBSET:
            data_list = self.data_train
        else:
            raise Exception("Invalid subset_name '%s'. Expected '%s', '%s', or '%s'." %
                            (subset_name, TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET))
        return data_list

    def get_augmented_numpy_subset(self, subset_name, mark_mode, border_sec):
        # Get augmented pages for random cropping.
        # border_sec: Seconds to be added at both borders of the augmented page.
        data_list = self.get_subset(subset_name)
        border_size = border_sec * self.fs
        features = []
        labels = []
        # Iterate over registers
        for i in range(len(data_list)):
            ind_dict = data_list[i]
            signal = ind_dict['signal']
            n2_pages = ind_dict['pages']
            if mark_mode == 1:
                marks = ind_dict['marks_1']
            elif mark_mode == 2:
                marks = ind_dict['marks_2']
            else:
                raise Exception("Invalid mark_mode %s. Expected 1 or 2." % mark_mode)
            # Iterate over pages
            for page in n2_pages:
                offset = page * self.page_size
                start_sample = int(offset - self.page_size/2 - border_size)
                end_sample = int(start_sample + 2*self.page_size + 2*border_size)
                augmented_page_signal = signal[start_sample:end_sample]
                augmented_page_labels = marks[start_sample:end_sample]
                features.append(augmented_page_signal)
                labels.append(augmented_page_labels)
        features_np = np.stack(features, axis=0)
        labels_np = np.stack(labels, axis=0)
        return features_np, labels_np
