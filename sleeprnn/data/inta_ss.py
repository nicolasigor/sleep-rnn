"""inta_ss.py: Defines the INTA class that manipulates the INTA database."""

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
from .dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS

PATH_INTA_RELATIVE = 'inta'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

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


class IntaSS(Dataset):
    """This is a class to manipulate the INTA data EEG dataset.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_INTA_RELATIVE
    |__ PATH_REC
        |__ ADGU101504.rec
        |__ ALUR012904.rec
        |__ ...
    |__ PATH_STATES
        |__ StagesOnly_ADGU101504.txt
        |__ ...
    |__ PATH_MARKS
        |__ NewerWinsFix_SS_ADGU101504.txt
        |__ ...

    If '...Fix...' marks files do not exist, then you should
    set the 'repair_stamps' flag to True. In that case, it is expected:

    |__ PATH_MARKS
        |__ SS_ADGU101504.txt
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False, repair_stamps=False):
        """Constructor"""
        # INTA parameters
        self.channel = 0  # Channel for SS, first is F4-C4, third is F3-C3
        self.n2_id = 3  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        # Sleep spindles characteristics
        self.min_ss_duration = 0.5  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        if repair_stamps:
            self._repair_stamps()

        valid_ids = [i for i in range(1, 12) if i not in IDS_INVALID]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]

        print('Train size: %d. Test size: %d'
              % (len(self.train_ids), len(self.test_ids)))
        print('Train subjects: \n', self.train_ids)
        print('Test subjects: \n', self.test_ids)

        super(IntaSS, self).__init__(
            dataset_dir=PATH_INTA_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.INTA_SS_NAME,
            all_ids=self.train_ids + self.test_ids,
            event_name=constants.SPINDLE,
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

            n2_pages = self._read_states(
                path_dict[KEY_FILE_STATES], signal_len)
            total_pages = int(np.ceil(signal_len / self.page_size))
            all_pages = np.arange(1, total_pages - 2, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])

            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS])
            print('Marks SS from E1: %d' % marks_1.shape[0])

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
            subject_name = NAMES[subject_id - 1]
            path_eeg_file = os.path.join(
                self.dataset_dir, PATH_REC,
                '%s.rec' % subject_name)
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES,
                'StagesOnly_%s.txt' % subject_name)
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                'NewerWinsFix_v2_SS_%s.txt' % subject_name)
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
        """Loads signal from 'path_eeg_file', does filtering."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            signal = file.readSignal(self.channel)
            fs_old = file.samplefrequency(self.channel)
            # Check
            print('Channel extracted: %s' % file.getLabel(self.channel))

        # Particular for INTA dataset
        fs_old = int(np.round(fs_old))

        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old)

        # Now resample to the required frequency
        if self.fs != fs_old:
            print('Resampling from %d Hz to required %d Hz' % (fs_old, self.fs))
            signal = utils.resample_signal(
                signal, fs_old=fs_old, fs_new=self.fs)
        else:
            print('Signal already at required %d Hz' % self.fs)

        signal = signal.astype(np.float32)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        # Recovery sample-stamps
        marks_file = np.loadtxt(path_marks_file, dtype='i', delimiter=' ')
        marks = marks_file[marks_file[:, 5] == self.channel + 1][:, [0, 1]]
        marks = np.round(marks).astype(np.int32)

        # Sample-stamps assume 200Hz sampling rate
        if self.fs != 200:
            print('Correcting marks from 200 Hz to %d Hz' % self.fs)
            # We need to transform the marks to the new sampling rate
            marks_time = marks.astype(np.float32) / 200.0
            # Transform to sample-stamps
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
        states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
        # These are pages with 30s durations. To work with 20s pages
        # We consider the intersections with 20s divisions
        n2_pages_original = np.where(states == self.n2_id)[0]
        print('Original N2 pages: %d' % n2_pages_original.size)
        onsets_original = n2_pages_original * self.original_page_duration
        offsets_original = (n2_pages_original + 1) * self.original_page_duration
        total_pages = int(np.ceil(signal_length / self.page_size))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int16)
        for i in range(total_pages):
            onset_new_page = i * self.page_duration
            offset_new_page = (i + 1) * self.page_duration
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
        n2_pages = n2_pages.astype(np.int16)
        return n2_pages

    def _repair_stamps(self):
        print('Repairing INTA stamps (Newer Wins Strategy + 0.5s criterion)')
        filename_format = 'NewerWinsFix_v2_SS_%s.txt'
        inta_folder = os.path.join(utils.PATH_DATA, PATH_INTA_RELATIVE)
        channel_for_txt = self.channel + 1
        for name in NAMES:
            print('Fixing %s' % name)
            path_marks_file = os.path.abspath(os.path.join(
                inta_folder, PATH_MARKS, 'SS_%s.txt' % name))
            path_eeg_file = os.path.abspath(os.path.join(
                inta_folder, PATH_REC, '%s.rec' % name))

            # Read marks
            print('Loading %s' % path_marks_file)
            data = np.loadtxt(path_marks_file)
            for_this_channel = data[:, -1] == channel_for_txt
            data = data[for_this_channel]
            data = np.round(data).astype(np.int32)

            # Remove zero duration marks, and ensure that start time < end time
            new_data = []
            for i in range(data.shape[0]):
                if data[i, 0] > data[i, 1]:
                    print('Start > End time found and fixed.')
                    aux = data[i, 0]
                    data[i, 0] = data[i, 1]
                    data[i, 1] = aux
                    new_data.append(data[i, :])
                elif data[i, 0] < data[i, 1]:
                    new_data.append(data[i, :])
                else:  # Zero duration (equality)
                    print('Zero duration mark found and removed')
            data = np.stack(new_data, axis=0)

            # Remove stamps outside signal boundaries
            print('Loading %s' % path_eeg_file)
            with pyedflib.EdfReader(path_eeg_file) as file:
                signal = file.readSignal(0)
                signal_len = signal.shape[0]
            new_data = []
            for i in range(data.shape[0]):
                if data[i, 1] < signal_len:
                    new_data.append(data[i, :])
                else:
                    print('Stamp outside boundaries found and removed')
            data = np.stack(new_data, axis=0)

            raw_marks = data[:, [0, 1]]
            valid = data[:, 4]

            print('Starting correction... ', flush=True)
            # Separate according to valid value. Valid = 0 is ignored.

            raw_marks_1 = raw_marks[valid == 1]
            raw_marks_2 = raw_marks[valid == 2]

            print('Originally: %d marks with valid=1, %d marks with valid=2'
                  % (len(raw_marks_1), len(raw_marks_2)))

            # Remove marks with duration less than 0.5s
            size_thr = 100
            print('Removing events with less than %d samples.' % size_thr)
            durations_1 = raw_marks_1[:, 1] - raw_marks_1[:, 0]
            raw_marks_1 = raw_marks_1[durations_1 >= size_thr]
            durations_2 = raw_marks_2[:, 1] - raw_marks_2[:, 0]
            raw_marks_2 = raw_marks_2[durations_2 >= size_thr]

            print('After duration criterion: %d marks with valid=1, %d marks with valid=2'
                  % (len(raw_marks_1), len(raw_marks_2)))

            # Now we add sequentially from the end (from newer marks), and we
            # only add marks if they don't intersect with the current set.
            # In this way, we effectively choose newer stamps over old ones
            # We start with valid=2, and then we continue with valid=1, to
            # follow the correction rule:
            # Keep valid=2 always
            # Keep valid=1 only if there is no intersection with valid=2

            n_v1 = 0
            n_v2 = 0

            if len(raw_marks_2) > 0:
                final_marks = [raw_marks_2[-1, :]]
                final_valid = [2]
                n_v2 += 1
                for i in range(raw_marks_2.shape[0] - 2, -1, -1):
                    candidate_mark = raw_marks_2[i, :]
                    current_set = np.stack(final_marks, axis=0)
                    if not utils.stamp_intersects_set(candidate_mark, current_set):
                        # There is no intersection
                        final_marks.append(candidate_mark)
                        final_valid.append(2)
                        n_v2 += 1
                for i in range(raw_marks_1.shape[0] - 1, -1, -1):
                    candidate_mark = raw_marks_1[i, :]
                    current_set = np.stack(final_marks, axis=0)
                    if not utils.stamp_intersects_set(candidate_mark,
                                                      current_set):
                        # There is no intersection
                        final_marks.append(candidate_mark)
                        final_valid.append(1)
                        n_v1 += 1
            else:
                print('There is no valid=2 marks.')
                final_marks = [raw_marks_1[-1, :]]
                final_valid = [1]
                n_v1 += 1
                for i in range(raw_marks_1.shape[0] - 2, -1, -1):
                    candidate_mark = raw_marks_1[i, :]
                    current_set = np.stack(final_marks, axis=0)
                    if not utils.stamp_intersects_set(candidate_mark,
                                                      current_set):
                        # There is no intersection
                        final_marks.append(candidate_mark)
                        final_valid.append(1)
                        n_v1 += 1

            print('Finally: %d with valid=1, %d with valid=2' % (n_v1, n_v2))

            # Now concatenate everything
            final_marks = np.stack(final_marks, axis=0)
            final_valid = np.stack(final_valid, axis=0)

            # And sort according to time
            idx_sorted = np.argsort(final_marks[:, 0])
            final_marks = final_marks[idx_sorted]
            final_valid = final_valid[idx_sorted]

            # Now create array in right format
            # [start end -50 -50 valid channel]

            number_for_txt = -50
            n_marks = final_marks.shape[0]
            channel_column = channel_for_txt * np.ones(n_marks).reshape(
                [n_marks, 1])
            number_column = number_for_txt * np.ones(n_marks).reshape(
                [n_marks, 1])
            valid_column = final_valid.reshape([n_marks, 1])
            table = np.concatenate(
                [final_marks,
                 number_column, number_column,
                 valid_column, channel_column],
                axis=1
            )
            table = table.astype(np.int32)

            # Now sort according to start time
            table = table[table[:, 0].argsort()]
            print('Done. %d marks for channel %d' % (n_marks, channel_for_txt))

            # Now save into a file
            path_new_marks_file = os.path.abspath(os.path.join(
                inta_folder, PATH_MARKS, filename_format % name))
            np.savetxt(path_new_marks_file, table, fmt='%d', delimiter=' ')
            print('Fixed marks saved at %s\n' % path_new_marks_file)

    # def _old_repair_stamps(self):
    #     print('Repairing INTA stamps (Fusion Strategy)')
    #     filename_format = 'FusionFix_SS_%s.txt'
    #     inta_folder = os.path.join(utils.PATH_DATA, PATH_INTA_RELATIVE)
    #     for name in NAMES:
    #         print('Fixing %s' % name)
    #         path_marks_file = os.path.abspath(os.path.join(
    #             inta_folder, PATH_MARKS, 'SS_%s.txt' % name))
    #         path_eeg_file = os.path.abspath(os.path.join(
    #             inta_folder, PATH_REC, '%s.rec' % name))
    #
    #         # Read marks
    #         print('Loading %s' % path_marks_file)
    #         data = np.loadtxt(path_marks_file)
    #         for_this_channel = data[:, -1] == self.channel + 1
    #         data = data[for_this_channel]
    #         data = np.round(data).astype(np.int32)
    #
    #         # Remove zero duration marks, and ensure that start time<end time
    #         new_data = []
    #         for i in range(data.shape[0]):
    #             if data[i, 0] > data[i, 1]:
    #                 aux = data[i, 0]
    #                 data[i, 0] = data[i, 1]
    #                 data[i, 1] = aux
    #                 new_data.append(data[i, :])
    #             elif data[i, 0] < data[i, 1]:
    #                 new_data.append(data[i, :])
    #             else:  # Zero duration (equality)
    #                 print('Zero duration mark found and removed')
    #         data = np.stack(new_data, axis=0)
    #
    #         raw_marks = data[:, [0, 1]]
    #         valid = data[:, 4]
    #
    #         print('Loading %s' % path_eeg_file)
    #         with pyedflib.EdfReader(path_eeg_file) as file:
    #             signal = file.readSignal(0)
    #             signal_len = signal.shape[0]
    #
    #         print('Starting correction... ', end='', flush=True)
    #         # Separate according to valid value. Valid = 0 is ignored.
    #         raw_marks_1 = raw_marks[valid == 1]
    #         raw_marks_2 = raw_marks[valid == 2]
    #
    #         # Turn into binary sequence
    #         raw_marks_1 = utils.stamp2seq(raw_marks_1, 0, signal_len - 1,
    #                                       allow_early_end=True)
    #         raw_marks_2 = utils.stamp2seq(raw_marks_2, 0, signal_len - 1,
    #                                       allow_early_end=True)
    #         # Go back to intervals
    #         raw_marks_1 = utils.seq2stamp(raw_marks_1)
    #         raw_marks_2 = utils.seq2stamp(raw_marks_2)
    #         # In this way, overlapping intervals are now together
    #
    #         # Correction rule:
    #         # Keep valid=2 always
    #         # Keep valid=1 only if there is no intersection with valid=2
    #         final_marks = [raw_marks_2]
    #         final_valid = [2 * np.ones(raw_marks_2.shape[0])]
    #         for i in range(raw_marks_1.shape[0]):
    #             # Check if there is any intersection
    #             add_condition = True
    #             for j in range(raw_marks_2.shape[0]):
    #                 start_intersection = max(raw_marks_1[i, 0],
    #                                          raw_marks_2[j, 0])
    #                 end_intersection = min(raw_marks_1[i,1], raw_marks_2[j,1])
    #                 intersection = end_intersection - start_intersection
    #                 if intersection >= 0:
    #                     add_condition = False
    #                     break
    #             if add_condition:
    #                 final_marks.append(raw_marks_1[[i], :])
    #                 final_valid.append([1])
    #
    #         # Now concatenate everything
    #         final_marks = np.concatenate(final_marks, axis=0)
    #         final_valid = np.concatenate(final_valid, axis=0)
    #
    #         # Now create array in right format
    #         # [start end -50 -50 valid channel]
    #         channel_for_txt = self.channel + 1
    #         number_for_txt = -50
    #         n_marks = final_marks.shape[0]
    #         channel_column = channel_for_txt * np.ones(n_marks).reshape(
    #             [n_marks, 1])
    #         number_column = number_for_txt * np.ones(n_marks).reshape(
    #             [n_marks, 1])
    #         valid_column = final_valid.reshape([n_marks, 1])
    #         table = np.concatenate(
    #             [final_marks,
    #              number_column, number_column,
    #              valid_column, channel_column],
    #             axis=1
    #         )
    #         table = table.astype(np.int32)
    #
    #         # Now sort according to start time
    #         table = table[table[:, 0].argsort()]
    #         print('Done')
    #
    #         # Now save into a file
    #         path_new_marks_file = os.path.abspath(os.path.join(
    #             inta_folder, PATH_MARKS, filename_format % name))
    #         np.savetxt(path_new_marks_file, table, fmt='%d', delimiter=' ')
    #         print('Fixed marks saved at %s\n' % path_new_marks_file)
