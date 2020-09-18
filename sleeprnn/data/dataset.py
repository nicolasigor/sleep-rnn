"""Class definition to manipulate data spindle EEG datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

from sleeprnn.common import pkeys
from sleeprnn.common import constants
from sleeprnn.common import checks
from . import utils

KEY_EEG = 'signal'
KEY_N2_PAGES = 'n2_pages'
KEY_ALL_PAGES = 'all_pages'
KEY_MARKS = 'marks'
KEY_HYPNOGRAM = 'hypnogram'


class Dataset(object):
    """This is a base class for data micro-events datasets.
    It provides the option to load and create checkpoints of the processed
    data, and provides methods to query data from specific ids or entire
    subsets.
    You have to overwrite the method '_load_from_files'.
    """

    def __init__(
            self,
            dataset_dir,
            load_checkpoint,
            dataset_name,
            all_ids,
            event_name,
            n_experts=1,
            params=None,
            verbose=True
    ):
        """Constructor.

        Args:
            dataset_dir: (String) Path to the folder containing the dataset.
               This path can be absolute, or relative to the project root.
            load_checkpoint: (Boolean). Whether to load from a checkpoint or to
               load from scratch using the original files of the dataset.
            dataset_name: (String) Name of the dataset. This name will be used for
               checkpoints.
            n_experts: (Int) Number of available data experts for data
                spindle annotations.
            all_ids: (list of int) List of available IDs.
        """
        # Save attributes
        if os.path.isabs(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = os.path.abspath(
                os.path.join(utils.PATH_DATA, dataset_dir))
        # We verify that the directory exists
        checks.check_directory(self.dataset_dir)

        self.load_checkpoint = load_checkpoint
        self.dataset_name = dataset_name
        self.event_name = event_name
        self.n_experts = n_experts
        self.ckpt_dir = os.path.abspath(os.path.join(
            self.dataset_dir, '..', 'ckpt_%s' % self.dataset_name))
        self.all_ids = all_ids
        self.all_ids.sort()
        if verbose:
            print('Dataset %s with %d patients.'
                  % (self.dataset_name, len(self.all_ids)))

        # events and data EEG related parameters
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults

        # Sampling frequency [Hz] to be used (not the original)
        self.fs = self.params[pkeys.FS]
        # Time of window page [s]
        self.page_duration = self.params[pkeys.PAGE_DURATION]
        self.page_size = int(self.page_duration * self.fs)

        # Ckpt file associated with the sampling frequency
        self.ckpt_file = os.path.join(
            self.ckpt_dir, '%s_fs%d.pickle' % (self.dataset_name, self.fs))

        # Data loading
        self.data = self._load_data(verbose=verbose)
        self.global_std = 1.0
        # FFT norm stuff
        self.mean_fft_scaling = 1.0

    def compute_global_std(self, subject_ids):
        x_list = self.get_subset_signals(
            subject_id_list=subject_ids, normalize_clip=False)
        tmp_list = []
        for x in x_list:
            outlier_thr = np.percentile(np.abs(x), 99)
            tmp_signal = x[np.abs(x) <= outlier_thr]
            tmp_list.append(tmp_signal)
        all_signals = np.concatenate(tmp_list)
        global_std = all_signals.std()
        return global_std

    def compute_fft_scaling_factor(self, band=[2, 6]):
        # Using FFT on whole page
        window_han = np.hanning(self.page_size)
        fft_scaling_factor_dict = {}
        for subject_id in self.all_ids:
            amp_all = []
            signal = self.get_subject_signal(subject_id, normalize_clip=False)
            n2_pages = self.data[subject_id][KEY_N2_PAGES]
            for page in n2_pages:
                start_page = page * self.page_size
                end_page = start_page + self.page_size
                amp, freq = utils.power_spectrum(window_han * signal[start_page:end_page], self.fs)
                amp_all.append(amp)
            amp_all = np.stack(amp_all, axis=0).mean(axis=0)
            band_low = (freq >= band[0]).astype(np.float32)
            band_high = (freq <= band[1]).astype(np.float32)
            band_weights = band_low * band_high
            band_mean = np.sum(amp_all * band_weights) / np.sum(band_weights)
            fft_scaling_factor_dict[subject_id] = 1 / band_mean
        return fft_scaling_factor_dict

    def get_subject_signal(
            self,
            subject_id,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=1,
            verbose=False
    ):
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            normalization_mode, 'normalization_mode',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        # Unpack data
        signal = ind_dict[KEY_EEG]

        if normalize_clip:
            if normalization_mode == constants.WN_RECORD:
                if verbose:
                    print('Normalization with stats from '
                          'pages containing true events.')
                # Normalize using stats from pages with true events.
                marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
                # Transform stamps into sequence
                marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = utils.extract_pages(
                    marks, tmp_pages,
                    self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal, tmp_pages, self.page_size, clip_value=self.params[pkeys.CLIP_VALUE],
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std, mean_fft_scaling=self.mean_fft_scaling)
            else:
                if verbose:
                    print('Normalization with stats from '
                          'N2 pages.')
                n2_pages = ind_dict[KEY_N2_PAGES]
                signal, _ = utils.norm_clip_signal(
                    signal, n2_pages, self.page_size, clip_value=self.params[pkeys.CLIP_VALUE],
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std, mean_fft_scaling=self.mean_fft_scaling)
        return signal

    def get_subset_signals(
            self,
            subject_id_list,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=1,
            verbose=False
    ):
        subset_signals = []
        for subject_id in subject_id_list:
            signal = self.get_subject_signal(
                subject_id,
                normalize_clip=normalize_clip,
                normalization_mode=normalization_mode,
                which_expert=which_expert,
                verbose=verbose)
            subset_signals.append(signal)
        return subset_signals

    def get_signals(
            self,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=1,
            verbose=False
    ):
        subset_signals = self.get_subset_signals(
            self.all_ids,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            which_expert=which_expert,
            verbose=verbose)
        return subset_signals

    def get_ids(self):
        return self.all_ids

    def get_subject_pages(
            self,
            subject_id,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the indices of the pages of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        if verbose:
            print('Getting ID %s, %d %s pages'
                  % (subject_id, pages.size, pages_subset))
        return pages

    def get_subset_pages(
            self,
            subject_id_list,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of pages from a list of subjects."""
        subset_pages = []
        for subject_id in subject_id_list:
            pages = self.get_subject_pages(
                subject_id,
                pages_subset=pages_subset,
                verbose=verbose)
            subset_pages.append(pages)
        return subset_pages

    def get_pages(
            self,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of pages from all subjects."""
        subset_pages = self.get_subset_pages(
            self.all_ids,
            pages_subset=pages_subset,
            verbose=verbose
        )
        return subset_pages

    def get_subject_stamps(
            self,
            subject_id,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the sample-stamps of marks of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]

        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        # Get stamps that are inside selected pages
        marks = utils.extract_pages_for_stamps(
            marks, pages, self.page_size)

        if verbose:
            print('Getting ID %s, %s pages, %d stamps'
                  % (subject_id, pages_subset, marks.shape[0]))
        return marks

    def get_subset_stamps(
            self,
            subject_id_list,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of stamps from a list of subjects."""
        subset_marks = []
        for subject_id in subject_id_list:
            marks = self.get_subject_stamps(
                subject_id,
                which_expert=which_expert,
                pages_subset=pages_subset,
                verbose=verbose)
            subset_marks.append(marks)
        return subset_marks

    def get_stamps(
            self,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of stamps from all subjects."""
        subset_marks = self.get_subset_stamps(
            self.all_ids,
            which_expert=which_expert,
            pages_subset=pages_subset,
            verbose=verbose
        )
        return subset_marks

    def get_subject_hypnogram(
            self,
            subject_id,
            verbose=False
    ):
        """Returns the hypogram of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)

        ind_dict = self.data[subject_id]

        hypno = ind_dict[KEY_HYPNOGRAM]

        if verbose:
            print('Getting Hypnogram of ID %s' % subject_id)
        return hypno

    def get_subset_hypnograms(
            self,
            subject_id_list,
            verbose=False
    ):
        """Returns the list of hypograms from a list of subjects."""
        subset_hypnos = []
        for subject_id in subject_id_list:
            hypno = self.get_subject_hypnogram(
                subject_id,
                verbose=verbose)
            subset_hypnos.append(hypno)
        return subset_hypnos

    def get_hypnograms(
            self,
            verbose=False
    ):
        """Returns the list of hypograms from all subjects."""
        subset_hypnos = self.get_subset_hypnograms(
            self.all_ids,
            verbose=verbose
        )
        return subset_hypnos

    def get_subject_data(
            self,
            subject_id,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False,
    ):
        """Returns segments of signal and marks from pages for the given id.

        Args:
            subject_id: (int) id of the subject of interest.
            augmented_page: (Optional, boolean, defaults to False) whether to
                augment the page with half page at each side.
            border_size: (Optional, int, defaults to 0) number of samples to be
                added at each border of the segments.
            forced_mark_separation_size: (Optional, int, defaults to 0) number
                of samples that are forced to exist between contiguous marks.
                If 0, no modification is performed.
            which_expert: (Optional, int, defaults to 1) Which expert
                annotations should be returned. It has to be consistent with
                the given n_experts, in a one-based counting.
            pages_subset: (Optional, string, [WN_RECORD, N2_RECORD]) If
                WN_RECORD (default), pages from the whole record. If N2_RECORD,
                only N2 pages are returned.
            normalize_clip: (Optional, boolean, defaults to True) If true,
                the signal is normalized and clipped from pages statistics.
            normalization_mode: (Optional, string, [WN_RECORD, N2_RECORD]) If
                WN_RECORD (default), statistics for normalization are
                computed from pages containing true events. If N2_RECORD,
                statistics are computed from N2 pages.
            verbose: (Optional, boolean, defaults to False) Whether to print
                what is being read.

        Returns:
            signal: (2D array) each row is an (augmented) page of the signal
            marks: (2D array) each row is an (augmented) page of the marks
        """
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i+1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])
        checks.check_valid_value(
            normalization_mode, 'normalization_mode',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        # Unpack data
        signal = ind_dict[KEY_EEG]
        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        # Transform stamps into sequence
        if forced_mark_separation_size > 0:
            print('Forcing separation of %d samples between marks' % forced_mark_separation_size)
            marks = utils.stamp2seq_with_separation(
                marks, 0, signal.shape[0] - 1, min_separation_samples=forced_mark_separation_size)
        else:
            marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)

        # Compute border to be added
        if augmented_page:
            total_border = self.page_size // 2 + border_size
        else:
            total_border = border_size

        if normalize_clip:
            if normalization_mode == constants.WN_RECORD:
                if True:  # verbose:
                    print('Normalization with stats from '
                          'pages containing true events.')
                # Normalize using stats from pages with true events.
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = utils.extract_pages(
                    marks, tmp_pages,
                    self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal, tmp_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std, clip_value=self.params[pkeys.CLIP_VALUE],
                    mean_fft_scaling=self.mean_fft_scaling)
            else:
                if verbose:
                    print('Normalization with stats from '
                          'N2 pages.')
                n2_pages = ind_dict[KEY_N2_PAGES]
                signal, _ = utils.norm_clip_signal(
                    signal, n2_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std, clip_value=self.params[pkeys.CLIP_VALUE],
                    mean_fft_scaling=self.mean_fft_scaling)

        # Extract segments
        signal = utils.extract_pages(
            signal, pages, self.page_size, border_size=total_border)
        marks = utils.extract_pages(
            marks, pages, self.page_size, border_size=total_border)

        if verbose:
            print('Getting ID %s, %d %s pages, Expert %d'
                  % (subject_id, pages.size, pages_subset, which_expert))
        return signal, marks

    def get_subset_data(
            self,
            subject_id_list,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False,
    ):
        """Returns the list of signals and marks from a list of subjects.
        """
        subset_signals = []
        subset_marks = []
        for subject_id in subject_id_list:
            signal, marks = self.get_subject_data(
                subject_id,
                augmented_page=augmented_page,
                border_size=border_size,
                forced_mark_separation_size=forced_mark_separation_size,
                which_expert=which_expert,
                pages_subset=pages_subset,
                normalize_clip=normalize_clip,
                normalization_mode=normalization_mode,
                verbose=verbose,
            )
            subset_signals.append(signal)
            subset_marks.append(marks)
        return subset_signals, subset_marks

    def get_data(
            self,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of signals and marks from all subjects.
        """
        subset_signals, subset_marks = self.get_subset_data(
            self.all_ids,
            augmented_page=augmented_page,
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            which_expert=which_expert,
            pages_subset=pages_subset,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            verbose=verbose
        )
        return subset_signals, subset_marks

    def get_sub_dataset(self, subject_id_list):
        """Data structure of a subset of subjects"""
        data_subset = {}
        for pat_id in subject_id_list:
            data_subset[pat_id] = self.data[pat_id].copy()
        return data_subset

    def save_checkpoint(self):
        """Saves a pickle file containing the loaded data."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(self.ckpt_file, 'wb') as handle:
            pickle.dump(
                self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Checkpoint saved at %s' % self.ckpt_file)

    def _load_data(self, verbose):
        """Loads data either from a checkpoint or from scratch."""
        if self.load_checkpoint and self._exists_checkpoint():
            if verbose:
                print('Loading from checkpoint... ', flush=True, end='')
            data = self._load_from_checkpoint()
        else:
            if verbose:
                if self.load_checkpoint:
                    print("A checkpoint doesn't exist at %s."
                          " Loading from source instead." % self.ckpt_file)
                else:
                    print('Loading from source.')
            data = self._load_from_source()
        if verbose:
            print('Loaded')
        return data

    def _load_from_checkpoint(self):
        """Loads the pickle file containing the loaded data."""
        with open(self.ckpt_file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def _exists_checkpoint(self):
        """Checks whether the pickle file with the checkpoint exists."""
        return os.path.isfile(self.ckpt_file)

    def _load_from_source(self):
        """Loads and return the data from files and transforms it appropriately.
        This is just a template for the specific implementation of the dataset.
        the value of the key KEY_ID has to be an integer.

        Signal is an 1D array, pages are indices, marks are 2D sample-stamps.
        """
        # Data structure
        data = {}
        for pat_id in self.all_ids:
            pat_dict = {
                KEY_EEG: None,
                KEY_N2_PAGES: None,
                KEY_ALL_PAGES: None,
                KEY_HYPNOGRAM: None
            }
            for i in range(self.n_experts):
                pat_dict.update(
                    {'%s_%d' % (KEY_MARKS, i+1): None})
            data[pat_id] = pat_dict
        return data

