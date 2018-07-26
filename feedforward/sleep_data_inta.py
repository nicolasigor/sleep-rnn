from __future__ import division
import numpy as np
import pyedflib
import time

import utils


class SleepDataINTA(object):

    def __init__(self):
        # Data params
        self.channel = 1           # Channel to be used
        self.dur_epoch = 30        # Time of window page [s]
        self.n2_val = 3            # N2 state coding value
        self.context = 1.28        # Length of context for timestep, in [s]
        self.percentile = 99       # Percentil for clipping
        self.fs = 200              # Sampling frequency of the dataset

        # Useful
        self.epoch_size = self.dur_epoch * self.fs
        # Split in train, val and test
        random_perm = [4, 7, 3, 8, 5, 6, 2, 0, 1, 9]
        test_idx = random_perm[0:2]
        val_idx = random_perm[2:4]
        train_idx = random_perm[4:]

        # Filenames params
        self.all_names = [
            'ADGU101504',
            'ALUR012904',
            # 'BECA011405',  we will skip this one for now
            'BRCA062405',
            'BRLO041102',
            'BTOL083105',
            'BTOL090105',
            'CAPO092605',
            'CRCA020205',
            'ESCI031905',
            'TAGO061203']

        path_rec = "ssdata/register/"
        rec_postamble = ".rec"

        path_marks = "ssdata/label/marks/"
        marks_preamble = "FixedSS_"
        marks_postamble = ".txt"

        path_states = "ssdata/label/states/"
        states_preamble = "StagesOnly_"
        states_postamble = ".txt"

        # Build list of paths
        data_path_list = []
        for i in range(len(self.all_names)):
            path_edf_file = path_rec + self.all_names[i] + rec_postamble
            path_states_file = path_states + states_preamble + self.all_names[i] + states_postamble
            path_marks_file = path_marks + marks_preamble + self.all_names[i] + marks_postamble
            # Save data
            ind_dict = {'file_edf': path_edf_file,
                        'file_states': path_states_file,
                        'file_marks': path_marks_file}
            data_path_list.append(ind_dict)
        print(len(data_path_list), ' records in INTA dataset.')

        train_path_list = [data_path_list[i] for i in train_idx]
        val_path_list = [data_path_list[i] for i in val_idx]
        test_path_list = [data_path_list[i] for i in test_idx]

        print('Training set size:', len(train_path_list), '-- Records ID:', train_idx)
        print('Validation set size:', len(val_path_list), '-- Records ID:', val_idx)
        print('Test set size:', len(test_path_list), '-- Records ID:', test_idx)

        # Load data
        print("Loading training set...")
        self.data_train = self.load_data(train_path_list)
        print("Loading validation set...")
        self.data_val = self.load_data(val_path_list)
        print("Loading test set...")
        self.data_test = self.load_data(test_path_list)

    def load_data(self, data_path_list):
        data_list = []
        n_data = len(data_path_list)
        start = time.time()
        for i in range(n_data):
            # Read EEG Signal, States, and Marks
            path_edf_file = data_path_list[i]['file_edf']
            path_states_file = data_path_list[i]['file_states']
            path_marks_file = data_path_list[i]['file_marks']
            signal, _ = self.read_eeg(path_edf_file, self.channel)
            states = self.read_states(path_states_file)
            marks = self.read_marks(path_marks_file, self.channel)
            # Find N2 epochs
            n2_epochs = self.get_n2epochs(states)
            # Clip-Normalize eeg signal
            signal = self.preprocessing_eeg(signal, n2_epochs)
            # Transform marks into 0_1 format
            marks = utils.inter2seq(marks, 0, signal.shape[0] - 1)
            # Save data
            ind_dict = {'signal': signal,
                        'epochs': n2_epochs,
                        'marks': marks}
            data_list.append(ind_dict)
            print(str(i+1) + '/' + str(n_data) + ' ready, time elapsed: ' + str(time.time() - start) + ' [s]')
        print(len(data_list), ' records have been read.')
        return data_list

    def read_eeg(self, path_edf_file, channel):
        file = pyedflib.EdfReader(path_edf_file)
        signal = file.readSignal(channel)
        fs = file.getSampleFrequency(channel)
        file._close()
        del file
        return signal, fs

    def read_marks(self, path_marks_file, channel):
        marks_file = np.loadtxt(path_marks_file, dtype='i', delimiter=' ')
        marks = marks_file[marks_file[:, 5] == channel][:, [0, 1]]
        return marks

    def read_states(self, path_states_file):
        states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
        # Source format is 1:SQ4  2:SQ3  3:SQ2  4:SQ1  5:REM  6:WA
        # We enforce the fusion of SQ3 and SQ4 in one single stage
        # So now 2:N3  3:N2  4:N1  5:R  6:W
        states[states == 1] = 2
        return states

    def get_n2epochs(self, states):
        n2_epochs = np.where(states == self.n2_val)[0]
        # Drop first and last epoch of the whole registers if they where selected
        last_state = states.shape[0] - 1
        n2_epochs = n2_epochs[(n2_epochs != 0) & (n2_epochs != last_state)]
        return n2_epochs

    def preprocessing_eeg(self, signal, n2_epochs):
        # Concatenate every n2 epoch
        n2_list = []
        for epoch in n2_epochs:
            sample_start = epoch * self.epoch_size
            sample_end = (epoch + 1) * self.epoch_size
            n2_signal = signal[sample_start:sample_end]
            n2_list.append(n2_signal)
        n2_signal = np.concatenate(n2_list, axis=0)
        # Compute robust mean and std for n2 epochs
        thr = np.percentile(np.abs(n2_signal), self.percentile)
        n2_signal[np.abs(n2_signal) > thr] = float('nan')
        data_mean = np.nanmean(n2_signal)
        data_std = np.nanstd(n2_signal)
        # Normalization and clipping
        new_signal = (np.clip(signal, -thr, thr) - data_mean) / data_std
        return new_signal

    def next_batch(self, batch_size, segment_size, mark_smooth, dataset="TRAIN"):
        # Select dataset
        if dataset == "VAL":
            data_list = self.data_val
        elif dataset == "TEST":
            data_list = self.data_test
        else:
            data_list = self.data_train
        # Initialize batch
        features = np.zeros((batch_size, segment_size))
        labels = np.zeros(batch_size)

        # FOR NOW THIS IS ALWAYS RANDOM

        # Choose registers
        n_data = len(data_list)
        ind_choice = np.random.choice(np.arange(n_data), batch_size, replace=True)
        for i in range(batch_size):
            # Select that register
            ind_dict = data_list[ind_choice[i]]
            # Choose a random epoch
            epoch = np.random.choice(ind_dict['epochs'])
            offset = epoch * self.epoch_size
            # Choose a random timestep in that epoch
            central_sample = np.random.choice(np.arange(self.epoch_size))
            central_sample = offset + central_sample
            # Get signal segment
            sample_start = central_sample - int(segment_size / 2)
            sample_end = central_sample + int(segment_size / 2)
            features[i, :] = ind_dict['signal'][sample_start:sample_end]
            # Get mark, with an optional smoothing
            smooth_start = central_sample - int(np.floor(mark_smooth / 2))
            smooth_end = smooth_start + mark_smooth
            mark_array = ind_dict['marks'][smooth_start:smooth_end]
            smooth_mark = np.mean(mark_array)
            labels[i] = smooth_mark
        return features, labels
