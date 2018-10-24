from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
import pyedflib

from . import data_ops


# TODO: update INTA class
class INTA(object):

    def __init__(self, load_checkpoint=False):
        self.name = "INTA"
        self.fs = 200               # Sampling frequency [Hz]
        self.dur_page = 30          # Time of window page [s]
        self.page_size = int(self.dur_page * self.fs)

        if load_checkpoint:
            print("\nLoading " + self.name + " from checkpoint")
            self.data_train = self.load_subset_checkpoint("train")
            self.data_val = self.load_subset_checkpoint("val")
            self.data_test = self.load_subset_checkpoint("test")
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
        print("\nPages in train set: " + str(n_pages_train))
        n_pages_val = np.sum([ind['pages'].shape[0] for ind in self.data_val])
        print("Pages in val set: " + str(n_pages_val))
        n_pages_test = np.sum([ind['pages'].shape[0] for ind in self.data_test])
        print("Pages in test set: " + str(n_pages_test))
        print("\nPages in " + self.name + " dataset: " + str(n_pages_train + n_pages_val + n_pages_test))

    def get_file_paths(self):
        all_names = [
            'ADGU101504',
            'ALUR012904',
            # 'BECA011405',  # we will skip this one for now
            'BRCA062405',
            'BRLO041102',
            'BTOL083105',
            'BTOL090105',
            'CAPO092605',
            'CRCA020205',
            'ESCI031905',
            'TAGO061203']

        path_rec = "sleep_data/ssdata_inta/register/"
        rec_postamble = ".rec"

        path_marks = "sleep_data/ssdata_inta/label/marks/"
        marks_preamble = "FixedSS_"
        marks_postamble = ".txt"

        path_states = "sleep_data/ssdata_inta/label/states/"
        states_preamble = "StagesOnly_"
        states_postamble = ".txt"

        # Build list of paths
        data_path_list = []
        for i in range(len(all_names)):
            path_eeg_file = path_rec + all_names[i] + rec_postamble
            path_states_file = path_states + states_preamble + all_names[i] + states_postamble
            path_marks_file = path_marks + marks_preamble + all_names[i] + marks_postamble
            # Save data
            ind_dict = {'file_eeg': path_eeg_file, 'file_states': path_states_file, 'file_marks': path_marks_file,
                        'reg_id': all_names[i]}
            data_path_list.append(ind_dict)
        print(len(data_path_list), 'records in ' + str(self.name) + ' dataset.')
        return data_path_list

    def random_split(self, data_path_list):
        random_perm = [3, 7, 2, 8, 5, 6, 4, 0, 1, 9]
        test_idx = random_perm[0:2]
        val_idx = random_perm[2:4]
        train_idx = random_perm[4:]

        train_path_list = [data_path_list[i] for i in train_idx]
        val_path_list = [data_path_list[i] for i in val_idx]
        test_path_list = [data_path_list[i] for i in test_idx]

        print('Train set size:', len(train_path_list), '-- Records ID:', train_idx)
        print('Val set size:', len(val_path_list), '-- Records ID:', val_idx)
        print('Test set size:', len(test_path_list), '-- Records ID:', test_idx)

        return train_path_list, val_path_list, test_path_list

    def save_checkpoint(self):
        self.save_subset_checkpoint(self.data_train, "train")
        self.save_subset_checkpoint(self.data_val, "val")
        self.save_subset_checkpoint(self.data_test, "test")

    def save_subset_checkpoint(self, data_list, subset_name):
        filename = "sleep_data/checkpoint_" + self.name + "/" + self.name + "_" + subset_name + ".pickle"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_subset_checkpoint(self, subset_name):
        filename = "sleep_data/checkpoint_" + self.name + "/" + self.name + "_" + subset_name + ".pickle"
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
            path_marks_file = data_path_list[i]['file_marks']
            signal = self.read_eeg(path_eeg_file)
            n2_pages = self.read_states(path_states_file)
            marks = self.read_marks(path_marks_file)
            # Clip-Normalize eeg signal
            signal = self.preprocessing_eeg(signal, n2_pages)
            # Transform marks into 0_1 format
            marks = data_ops.inter2seq(marks, 0, signal.shape[0] - 1)
            # Save data
            reg_id = data_path_list[i]['reg_id']
            ind_dict = {'signal': signal, 'pages': n2_pages, 'marks': marks, 'reg_id': reg_id}
            data_list.append(ind_dict)
            print(str(i+1) + '/' + str(n_data) + ' ready, time elapsed: ' + str(time.time() - start) + ' [s]')
        print(len(data_list), ' records have been read.')
        return data_list

    def read_eeg(self, path_eeg_file):
        channel = 1
        file = pyedflib.EdfReader(path_eeg_file)
        signal = file.readSignal(channel)
        # fs = file.getSampleFrequency(channel)
        file._close()
        del file
        return signal

    def read_marks(self, path_marks_file):
        channel = 1
        marks_file = np.loadtxt(path_marks_file, dtype='i', delimiter=' ')
        marks = marks_file[marks_file[:, 5] == channel][:, [0, 1]]
        return marks

    def read_states(self, path_states_file):
        states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
        # Source format is 1:SQ4  2:SQ3  3:SQ2  4:SQ1  5:REM  6:WA
        n2_val = 3
        n2_pages = np.where(states == n2_val)[0]
        # Drop first, last and second to last page of the whole registers if they where selected
        last_page = states.shape[0] - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page) & (n2_pages != last_page - 1)]
        return n2_pages

    def preprocessing_eeg(self, signal, n2_pages):
        # Concatenate every n2 page
        n2_list = []
        for page in n2_pages:
            sample_start = page * self.page_size
            sample_end = (page + 1) * self.page_size
            n2_signal = signal[sample_start:sample_end]
            n2_list.append(n2_signal)
        n2_signal = np.concatenate(n2_list, axis=0)
        # Compute robust mean and std for n2 pages
        thr = np.percentile(np.abs(n2_signal), 99)
        n2_signal[np.abs(n2_signal) > thr] = float('nan')
        data_mean = np.nanmean(n2_signal)
        data_std = np.nanstd(n2_signal)
        # Normalization and clipping
        new_signal = (np.clip(signal, -thr, thr) - data_mean) / data_std
        return new_signal

    def get_subset(self, subset_name):
        # Select subset
        if subset_name == "val":
            data_list = self.data_val
        elif subset_name == "test":
            data_list = self.data_test
        else:
            data_list = self.data_train
        return data_list

    def get_fs(self):
        return self.fs

    def get_page_size(self):
        return self.page_size

    def sample_batch(self, batch_size, segment_size, subset_name="train"):
        # Select subset
        data_list = self.get_subset(subset_name)
        # Initialize batch
        features = np.zeros((batch_size, 1, segment_size, 1), dtype=np.float32)
        labels = np.zeros(batch_size, dtype=np.float32)
        # Choose registers
        n_data = len(data_list)
        ind_choice = np.random.choice(np.arange(n_data), batch_size, replace=True)
        for i in range(batch_size):
            # Select that register
            ind_dict = data_list[ind_choice[i]]
            # Choose a random page
            epoch = np.random.choice(ind_dict['pages'])
            offset = epoch * self.page_size
            # Choose a random timestep in that page
            central_sample = np.random.choice(np.arange(self.page_size))
            central_sample = offset + central_sample
            # Get signal segment
            sample_start = central_sample - int(segment_size / 2)
            sample_end = central_sample + int(segment_size / 2)
            features[i, 0, :, 0] = ind_dict['signal'][sample_start:sample_end]
            # Get mark
            labels[i] = ind_dict['marks'][central_sample]
        return features, labels

    def sample_element(self, segment_size, subset_name="train"):
        # Select subset
        data_list = self.get_subset(subset_name)
        # Initialize element
        feature = np.zeros((1, segment_size, 1))
        # Choose a register
        n_data = len(data_list)
        ind_choice = np.random.choice(np.arange(n_data))
        # Select that register
        ind_dict = data_list[ind_choice]
        # Choose a random page
        epoch = np.random.choice(ind_dict['pages'])
        offset = epoch * self.page_size
        # Choose a random timestep in that page
        central_sample = np.random.choice(np.arange(self.page_size))
        central_sample = offset + central_sample
        # Get signal segment
        sample_start = central_sample - int(segment_size / 2)
        sample_end = central_sample + int(segment_size / 2)
        feature[0, :, 0] = ind_dict['signal'][sample_start:sample_end]
        # Get mark
        label = ind_dict['marks'][central_sample]
        # Fix type
        feature = np.array(feature, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        return feature, label
