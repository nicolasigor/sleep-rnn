"""isruc_ss.py: Defines the ISRUC class that manipulates the ISRUC database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd
import pyedflib

from sleeprnn.common import constants
from . import utils
from . import stamp_correction
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS

PATH_ISRUC_RELATIVE = 'unlabeled_isruc'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

# 20-42 y.o. subset, without 1-006 and 1-025.
IDS_VALID = [
    '1-079',
    '1-092',
    '1-048',
    '1-027',
    '1-069',
    '1-004',
    '1-046',
    '1-031',
    '1-057',
    '1-058',
    '1-050',
    '1-052',
    '1-053',
    '1-065',
    '1-033',
    '1-072',
    '1-087',
    '1-067',
    '1-090',
    '1-036',
    '1-099',
    '1-038',
    '1-003',
    '1-018',
    '1-093',
    '1-076',
    '1-098',
    '1-029',
    '1-062',
    '3-001',
    '3-007',
    '3-008',
    '3-009',
    '3-010',
    '3-006',
    '3-002',
]


class IsrucSS(Dataset):
    """This is a class to manipulate the ISRUC data EEG dataset.

    The sleep spindle marks are detections made by the A7 algorithm:
    Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., & Warby, S. C. (2019).
    A sleep spindle detection algorithm that emulates human expert spindle scoring.
    Journal of neuroscience methods, 316, 3-11.
    The four parameters were fitted on MASS-SS2.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_ISRUC_RELATIVE
    |__ PATH_REC
        |__ 1-003 PSG.rec
        |__ 1-004 PSG.rec
        |__ ...
    |__ PATH_STATES
        |__ 1-003 Base E1.txt
        |__ ...
    |__ PATH_MARKS
        |__ EventDetection_s1-079_*.txt
        |__ ...
    """

    def __init__(
            self, params=None, load_checkpoint=False, verbose=True, external_global_std=16.482042):
        """Constructor"""
        # ISRUC parameters
        self.channels_to_try = [
            ("C3-A2",),
            ("C3-M2",),
            ("C3", "A2"),
            ("C3", "M2")]  # Channel for SS
        self.n2_id = 2  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        # Sleep spindles characteristics
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        self.train_ids = IDS_VALID
        self.train_ids.sort()

        if verbose:
            print('Train size: %d.' % len(self.train_ids))
            print('Train subjects: \n', self.train_ids)

        super(IsrucSS, self).__init__(
            dataset_dir=PATH_ISRUC_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.ISRUC_SS_NAME,
            all_ids=self.train_ids,
            event_name=constants.SPINDLE,
            params=params,
            verbose=verbose
        )

        if external_global_std is not None:
            self.global_std = external_global_std
            self._scale_signals()

    def _scale_signals(self):
        print("Scaling signals to match global std %s" % self.global_std)
        for subject_id in self.all_ids:
            old_signal = self.data[subject_id][KEY_EEG]
            # Robust signal standard deviation normalization
            pctl_to_std = 98
            outlier_thr = np.percentile(np.abs(old_signal), pctl_to_std)
            tmp_signal = old_signal[np.abs(old_signal) <= outlier_thr]
            std_to_normalize = tmp_signal.std()
            # Scale signal to copy external global std
            new_signal = old_signal * self.global_std / std_to_normalize
            # Set new signal to dataset
            self.data[subject_id][KEY_EEG] = new_signal

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print('\nLoading ID %s' % subject_id)
            path_dict = data_paths[subject_id]

            # Read data
            signal = self._read_eeg(path_dict[KEY_FILE_EEG])
            signal_len = signal.shape[0]

            n2_pages = self._read_states(path_dict[KEY_FILE_STATES], signal_len)
            total_pages = int(np.ceil(signal_len / self.page_size))
            all_pages = np.arange(1, total_pages - 2, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])

            marks_1 = self._read_marks(path_dict['%s_1' % KEY_FILE_MARKS])
            print('Marks SS from A7: %d' % marks_1.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: marks_1
            }
            data[subject_id] = ind_dict
            print('Loaded ID %s (%02d/%02d ready). Time elapsed: %1.4f [s]' % (
                subject_id, i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_file = os.path.join(self.dataset_dir, PATH_REC, "%s PSG.rec" % subject_id)
            path_states_file = os.path.join(self.dataset_dir, PATH_STATES, "%s Base E1.txt" % subject_id)
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                'EventDetection_s%s_absSigPow(1.75)_relSigPow(1.6)_sigCov(1.8)_sigCorr(0.75).txt' % subject_id)
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
        channels_to_try = self.channels_to_try.copy()

        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            while len(channels_to_try) > 0:
                channel = channels_to_try.pop(0)
                if np.all([chn in channel_names for chn in channel]):
                    break
            chn = channel_names.index(channel[0])
            signal = file.readSignal(chn)
            fs_old = file.samplefrequency(chn)
            if len(channel) == 2:
                chn2 = channel_names.index(channel[1])
                signal2 = file.readSignal(chn2)
                signal = signal - signal2
                # Check
                print('Channel extracted: %s minus %s' % (file.getLabel(chn), file.getLabel(chn2)))
            else:
                # Check
                print('Channel extracted: %s' % file.getLabel(chn))

        # fs for ISRUC will be considered 200 Hz since it is a difference only at the 15th decimal.
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
        pred_data = pd.read_csv(path_marks_file, sep='\t')
        # We substract 1 to translate from matlab to numpy indexing system
        start_samples = pred_data.start_sample.values - 1
        end_samples = pred_data.end_sample.values - 1
        marks = np.stack([start_samples, end_samples], axis=1).astype(np.int32)

        # Sample-stamps assume 200Hz sampling rate
        if self.fs != 200:
            print('Correcting marks from 200 Hz to %d Hz' % self.fs)
            # We need to transform the marks to the new sampling rate
            marks_time = marks.astype(np.float32) / 200.0
            # Transform to sample-stamps
            marks = np.round(marks_time * self.fs).astype(np.int32)

        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(marks, self.fs, self.min_ss_duration, self.max_ss_duration)
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
                intersection = (onset_new_page < offsets_original[j]) and (onsets_original[j] < offset_new_page)
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
