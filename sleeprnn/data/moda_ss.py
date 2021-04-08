"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from sleeprnn.common import constants
from . import utils
from . import stamp_correction
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_MARKS
from .dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM

PATH_MODA_RELATIVE = 'moda'
PATH_SEGMENTS = 'segments/moda_preprocessed_segments.npz'

KEY_N_BLOCKS = 'n_blocks'
KEY_PHASE = 'phase'


class ModaSS(Dataset):
    def __init__(self, params=None, load_checkpoint=False, verbose=True):
        """Constructor"""
        self.original_fs = 256  # Hz
        self.original_border_duration = 30  # s
        # Hypnogram parameters
        self.unknown_id = '?'  # Character for unknown state in hypnogram
        self.n2_id = '2'  # Character for N2 identification in hypnogram

        # Sleep spindles characteristics
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        all_ids = self._get_ids()

        super(ModaSS, self).__init__(
            dataset_dir=PATH_MODA_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.MODA_SS_NAME,
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            n_experts=1,
            params=params,
            verbose=verbose,
        )

        self.global_std = None
        if verbose:
            print('Global STD:', self.global_std)

    def _get_ids(self):
        fpath = self._get_data_path()
        dataset = np.load(fpath)
        subjects_of_segments = dataset['subjects']
        subject_ids = np.unique(subjects_of_segments)
        subject_ids = subject_ids.tolist()
        subject_ids.sort()
        return subject_ids

    def _get_data_path(self):
        return os.path.abspath(os.path.join(utils.PATH_DATA, PATH_MODA_RELATIVE, PATH_SEGMENTS))

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        fpath = self._get_data_path()
        dataset = np.load(fpath)
        data = {}
        n_subjects = len(self.all_ids)
        start = time.time()
        for i, subject_id in enumerate(self.all_ids):
            print('\nLoading ID %s' % subject_id)
            subject_locs = np.sort(np.where(dataset['subjects'] == subject_id)[0])
            n_blocks = subject_locs.size
            phase = dataset['phases'][subject_locs[0]]
            signals = dataset['signals'][subject_locs, :]
            labels = dataset['labels'][subject_locs, :]
            signals = self._prepare_signals(signals)  # [n_samples]
            print("Debug:", n_blocks, "blocks, final shape of signal", signals.shape, signals.dtype)
            labels = self._prepare_labels(labels)  # [n_spindles, 2]
            print("Debug:", n_blocks, "blocks, final shape of labels", labels.shape, labels.dtype)
            n2_pages, hypnogram = self._generate_states(n_blocks)
            total_pages = hypnogram.size
            all_pages = np.arange(1, total_pages - 1, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Hypnogram pages: %d' % hypnogram.shape[0])
            print('Marks SS from E1: %d' % labels.shape[0])
            # Save data
            ind_dict = {
                KEY_EEG: signals,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: labels,
                KEY_HYPNOGRAM: hypnogram,
                KEY_PHASE: phase,
                KEY_N_BLOCKS: n_blocks
            }
            data[subject_id] = ind_dict
            print('Loaded ID %s (%03d/%03d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, i+1, n_subjects, time.time()-start))
        print('%d records have been read.' % len(data))
        return data

    def _prepare_signals(self, list_of_segments):
        # Each segment is of length 30s + 115s + 30s
        # We add 2.5s at each side of the blocks to make them of 120s = 6 * 20s
        # Therefore, each segment contributes with six 20s pages.
        # Additionally, we add 20s of border at each block to allow context.
        # Therefore, each segment contributes with two 20s pages of "?" state.
        # In summary, we need 20s + 120s + 20s = 22.5s + 115s + 22.5s
        target_border_duration = 22.5
        crop_size = int(self.original_fs * (self.original_border_duration - target_border_duration))
        list_of_segments = list_of_segments[:, crop_size:-crop_size]
        signal = np.concatenate(list_of_segments)
        # Now resample to the required frequency
        if self.fs != self.original_fs:
            print('Resampling from %d Hz to required %d Hz' % (self.original_fs, self.fs))
            signal = utils.resample_signal(signal, fs_old=self.original_fs, fs_new=self.fs)
        else:
            print('Signal already at required %d Hz' % self.fs)
        signal = signal.astype(np.float32)
        return signal

    def _prepare_labels(self, list_of_labels):
        target_border_duration = 22.5
        crop_size = int(self.original_fs * (self.original_border_duration - target_border_duration))
        list_of_labels = list_of_labels[:, crop_size:-crop_size]
        labels = np.concatenate(list_of_labels)
        binary_labels = np.clip(labels, a_min=0, a_max=1)  # borders will have a label of zero
        marks = utils.seq2stamp(binary_labels)
        marks_time = marks / self.original_fs  # sample to seconds
        # Transforms to sample-stamps
        marks = np.round(marks_time * self.fs).astype(np.int32)  # second to samples in target fs
        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _generate_states(self, n_blocks):
        # Each block has ?, 6x N2, and ?
        single_block = [self.unknown_id] + 6 * [self.n2_id] + [self.unknown_id]
        hypnogram = n_blocks * single_block
        hypnogram = np.asarray(hypnogram)
        # Extract N2 pages
        n2_pages = np.where(hypnogram == self.n2_id)[0]
        n2_pages = n2_pages.astype(np.int16)
        return n2_pages, hypnogram
