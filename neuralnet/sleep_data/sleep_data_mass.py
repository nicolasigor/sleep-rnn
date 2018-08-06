from __future__ import division
from __future__ import print_function
import numpy as np
import pyedflib
import time
import pickle
import os
import scipy.signal as sp_signal
import math

from sleep_data import utils


class SleepDataMASS(object):

    def __init__(self, load_from_checkpoint=False):
        self.name = "MASS"
        self.fs = 200               # Sampling frequency [Hz]
        self.dur_page = 20          # Time of window page [s]
        self.page_size = int(self.dur_page * self.fs)

        if load_from_checkpoint:
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
        # Only those registers that have marks from two experts
        reg_ids = ['01', '02', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '17', '18', '19']

        path_rec = "sleep_data/ssdata_mass/register/"
        rec_preamble = "01-02-00"
        rec_postamble = " PSG.edf"

        path_marks_1 = "sleep_data/ssdata_mass/label/marks/e1/"
        marks_1_preamble = "01-02-00"
        marks_1_postamble = " SpindleE1.edf"

        path_marks_2 = "sleep_data/ssdata_mass/label/marks/e2/"
        marks_2_preamble = "01-02-00"
        marks_2_postamble = " SpindleE2.edf"

        path_states = "sleep_data/ssdata_mass/label/states/"
        states_preamble = "01-02-00"
        states_postamble = " Base.edf"

        # Build list of paths
        data_path_list = []
        for i in range(len(reg_ids)):
            path_eeg_file = path_rec + rec_preamble + reg_ids[i] + rec_postamble
            path_states_file = path_states + states_preamble + reg_ids[i] + states_postamble
            path_marks_1_file = path_marks_1 + marks_1_preamble + reg_ids[i] + marks_1_postamble
            path_marks_2_file = path_marks_2 + marks_2_preamble + reg_ids[i] + marks_2_postamble
            # Save data
            ind_dict = {'file_eeg': path_eeg_file, 'file_states': path_states_file,
                        'file_marks_1': path_marks_1_file, 'file_marks_2': path_marks_2_file,
                        'reg_id': reg_ids[i]}
            data_path_list.append(ind_dict)
        print(len(data_path_list), 'records in ' + str(self.name) + ' dataset.')
        return data_path_list

    def random_split(self, data_path_list):
        random_perm = [13,  1,  4,  9,  2,  8, 10,  5, 11,  6,  3, 14, 12,  0,  7]
        test_idx = random_perm[0:4]
        val_idx = random_perm[4:7]
        train_idx = random_perm[7:]

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
            path_marks_1_file = data_path_list[i]['file_marks_1']
            path_marks_2_file = data_path_list[i]['file_marks_2']

            signal, signal_duration = self.read_eeg(path_eeg_file)
            n2_pages = self.read_states(path_states_file, signal.shape[0])
            marks_1 = self.read_marks(path_marks_1_file, signal.shape[0]/signal_duration)
            marks_2 = self.read_marks(path_marks_2_file, signal.shape[0]/signal_duration)

            # Clip-Normalize eeg signal
            signal = self.preprocessing_eeg(signal, n2_pages)
            # Transform marks into 0_1 format
            marks_1 = utils.inter2seq(marks_1, 0, signal.shape[0] - 1)
            marks_2 = utils.inter2seq(marks_2, 0, signal.shape[0] - 1)
            # Save data
            reg_id = data_path_list[i]['reg_id']
            ind_dict = {'signal': signal, 'pages': n2_pages,
                        'marks_1': marks_1, 'marks_2': marks_2,
                        'reg_id': reg_id}
            data_list.append(ind_dict)
            print(str(i+1) + '/' + str(n_data) + ' ready, time elapsed: ' + str(time.time() - start) + ' [s]')
        print(len(data_list), ' records have been read.')
        return data_list

    def read_eeg(self, path_eeg_file):
        channel = 13  # C3 in 13, F3 in 22
        file = pyedflib.EdfReader(path_eeg_file)
        signal = file.readSignal(channel)
        fs_old = file.getSampleFrequency(channel)
        signal_duration = file.file_duration
        file._close()
        del file
        # Resample
        gcd_freqs = math.gcd(self.fs, fs_old)
        up = int(self.fs / gcd_freqs)
        down = int(fs_old / gcd_freqs)
        signal = sp_signal.resample_poly(signal, up, down)
        signal = np.array(signal, dtype=np.float32)
        return signal, signal_duration

    def read_marks(self, path_marks_file, local_fs):
        file = pyedflib.EdfReader(path_marks_file)
        annotations = file.readAnnotations()
        file._close()
        del file
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        # Translate to a sample step:
        start_samples = np.array(np.round(onsets * local_fs), dtype=np.int32)
        end_samples = np.array(np.round(offsets * local_fs), dtype=np.int32)
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
            if stages_char[i] == '2':
                page_idx = int(np.round(onsets[i] / self.dur_page))
                n2_pages_onehot[page_idx] = 1
        n2_pages = np.where(n2_pages_onehot == 1)[0]
        # Drop first, last and second to last page of the whole registers if they where selected
        last_page = total_pages - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page) & (n2_pages != last_page - 1)]
        n2_pages = np.array(n2_pages, dtype=np.int32)
        return n2_pages

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
        if subset_name == "val":
            data_list = self.data_val
        elif subset_name == "test":
            data_list = self.data_test
        elif subset_name == "train":
            data_list = self.data_train
        else:
            raise Exception("Invalid subset_name. Expected 'train', 'val', or 'test'.")
        return data_list

    def get_fs(self):
        return self.fs

    def get_page_size(self):
        return self.page_size

    def get_page_duration(self):
        return self.dur_page

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
                raise Exception("Invalid mark_mode. Expected 1 or 2.")
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
