"""Class definition to manipulate data spindle EEG datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

from sleep.utils import checks
from . import data_ops
from .data_ops import PATH_DATA

KEY_EEG = 'signal'
KEY_USEFUL_PAGES = 'useful_pages'
KEY_ALL_PAGES = 'all_pages'
KEY_MARKS = 'marks'


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
            name,
            train_ids,
            test_ids,
            n_experts=1,
            fs=200,
            page_duration=20
    ):
        """Constructor.

        Args:
            dataset_dir: (String) Path to the folder containing the dataset.
               This path can be absolute, or relative to the project root.
            load_checkpoint: (Boolean). Whether to load from a checkpoint or to
               load from scratch using the original files of the dataset.
            name: (String) Name of the dataset. This name will be used for
               checkpoints.
            n_experts: (Int) Number of available data experts for data
                spindle annotations.
            train_ids: (list of int) List of IDs corresponding to the train
                subset subjects.
            test_ids: (list of int) List of IDs corresponding to the test
                subset subjects.
        """
        # Save attributes
        if os.path.isabs(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = os.path.join(PATH_DATA, dataset_dir)
        # We verify that the directory exists
        checks.check_directory(self.dataset_dir)

        self.load_checkpoint = load_checkpoint
        self.name = name
        self.n_experts = n_experts
        self.ckpt_dir = os.path.abspath(os.path.join(
            self.dataset_dir, '..', 'ckpt_%s' % self.name))
        self.ckpt_file = os.path.join(
            self.ckpt_dir, '%s.pickle' % self.name)
        self.train_ids = list(train_ids)
        self.test_ids = list(test_ids)
        self.all_ids = self.train_ids + self.test_ids
        self.train_ids.sort()
        self.test_ids.sort()
        self.all_ids.sort()
        print('Dataset %s with %d patients.' % (self.name, len(self.all_ids)))
        print('Train size: %d. Test size: %d'
              % (len(self.train_ids), len(self.test_ids)))
        print('Train subjects: \n', self.train_ids)
        print('Test subjects: \n', self.test_ids)

        # events and data EEG related parameters
        self.fs = fs  # Sampling frequency [Hz] to be used (not the original)
        self.page_duration = page_duration  # Time of window page [s]
        self.page_size = int(self.page_duration * self.fs)

        # Data loading
        self.data = self._load_data()

    def get_subject_pages(
            self,
            subject_id,
            whole_night=False,
            verbose=False):
        """Returns the indices of the pages of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        ind_dict = self.data[subject_id]

        if whole_night:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_USEFUL_PAGES]

        if verbose:
            if whole_night:
                descriptor = 'whole-night'
            else:
                descriptor = 'N2'
            print('Getting ID %s, %d %s pages'
                  % (subject_id, pages.size, descriptor))
        return pages

    def get_subset_pages(
            self,
            subject_id_list,
            whole_night=False,
            verbose=False
    ):
        """Returns the list of pages from a list of subjects."""
        subset_pages = []
        for subject_id in subject_id_list:
            pages = self.get_subject_pages(
                subject_id, whole_night, verbose)
            subset_pages.append(pages)
        return subset_pages

    def get_subject_stamps(
            self,
            subject_id,
            which_expert=1,
            whole_night=False,
            verbose=False):
        """Returns the sample-stamps of marks of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)

        ind_dict = self.data[subject_id]

        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]

        if whole_night:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_USEFUL_PAGES]

        # Get stamps that are inside selected pages
        marks = data_ops.extract_pages_with_stamps(
            marks, pages, self.page_size)

        if verbose:
            if whole_night:
                descriptor = 'whole-night'
            else:
                descriptor = 'N2'
            print('Getting ID %s, %s pages, %d stamps'
                  % (subject_id, descriptor, marks.shape[0]))
        return marks

    def get_subset_stamps(
            self,
            subject_id_list,
            which_expert=1,
            whole_night=False,
            verbose=False
    ):
        """Returns the list of stamps from a list of subjects."""
        subset_marks = []
        for subject_id in subject_id_list:
            marks = self.get_subject_stamps(
                subject_id, which_expert, whole_night, verbose)
            subset_marks.append(marks)
        return subset_marks

    def get_subject_data(
            self,
            subject_id,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            whole_night=False,
            normalize_clip=True,
            debug_force_n2stats=False,
            debug_force_activitystats=False,
            verbose=False
    ):
        """Returns segments of signal and marks from pages for the given id.

        Args:
            subject_id: (int) id of the subject of interest.
            augmented_page: (Optional, boolean, defaults to False) whether to
                augment the page with half page at each side.
            border_size: (Optional, int, defaults to 0) number of samples to be
                added at each border of the segments.
            which_expert: (Optional, int, defaults to 1) Which expert
                annotations should be returned. It has to be consistent with
                the given n_experts, in a one-based counting.
            whole_night: (Optional, boolean, defaults to False) If true, returns
                pages from the whole record. If false, only N2 pages are used.
            normalize_clip: (Optional, boolean, defaults to True) If true,
                the signal is normalized and clipped from pages statistics.
            verbose: (Optional, boolean, defaults to False) Whether to print
                what is being read.

        Returns:
            signal: (2D array) each row is an (augmented) page of the signal
            marks: (2D array) each row is an (augmented) page of the marks
        """
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i+1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)

        ind_dict = self.data[subject_id]

        # Unpack data
        signal = ind_dict[KEY_EEG]
        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
        if whole_night:
            descriptor = 'whole-night'
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            descriptor = 'N2'
            pages = ind_dict[KEY_USEFUL_PAGES]

        # Transform stamps into sequence
        marks = data_ops.inter2seq(marks, 0, signal.shape[0] - 1)

        # Compute border to be added
        if augmented_page:
            total_border = self.page_size // 2 + border_size
        else:
            total_border = border_size

        if normalize_clip:
            if debug_force_activitystats:
                print('Forcing activity normalization')
                # Normalize using stats from pages with true events.
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = data_ops.extract_pages(
                    marks, tmp_pages,
                    self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal = data_ops.norm_clip_eeg(
                    signal, tmp_pages, self.page_size)
            else:
                if debug_force_n2stats:
                    print('Forcing N2 normalization')
                    n2_pages = ind_dict[KEY_USEFUL_PAGES]
                    signal = data_ops.norm_clip_eeg(signal, n2_pages, self.page_size)
                else:
                    signal = data_ops.norm_clip_eeg(signal, pages, self.page_size)

        # Extract segments
        signal = data_ops.extract_pages(
            signal, pages, self.page_size, border_size=total_border)
        marks = data_ops.extract_pages(
            marks, pages, self.page_size, border_size=total_border)

        if verbose:
            print('Getting ID %s, %d %s pages, Expert %d'
                  % (subject_id, pages.size, descriptor, which_expert))
        return signal, marks

    def get_subset_data(
            self,
            subject_id_list,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            whole_night=False,
            normalize_clip=True,
            debug_force_n2stats=False,
            debug_force_activitystats=False,
            verbose=False
    ):
        """Returns the list of signals and marks from a list of subjects.
        """
        subset_signals = []
        subset_marks = []
        for subject_id in subject_id_list:
            signal, marks = self.get_subject_data(
                subject_id,
                augmented_page,
                border_size,
                which_expert,
                whole_night,
                normalize_clip,
                debug_force_n2stats,
                debug_force_activitystats,
                verbose)
            subset_signals.append(signal)
            subset_marks.append(marks)
        return subset_signals, subset_marks

    def save_checkpoint(self):
        """Saves a pickle file containing the loaded data."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(self.ckpt_file, 'wb') as handle:
            pickle.dump(
                self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Checkpoint saved at %s' % self.ckpt_file)

    def _load_data(self):
        """Loads data either from a checkpoint or from scratch."""
        if self.load_checkpoint and self._exists_checkpoint():
            print('Loading from checkpoint... ', flush=True, end='')
            data = self._load_from_checkpoint()
        else:
            if self.load_checkpoint:
                print("A checkpoint doesn't exist at %s."
                      " Loading from files instead." % self.ckpt_file)
            else:
                print('Loading from files.')
            data = self._load_from_files()
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

    def _load_from_files(self):
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
                KEY_USEFUL_PAGES: None,
                KEY_ALL_PAGES: None}
            for i in range(self.n_experts):
                pat_dict.update(
                    {'%s_%d' % (KEY_MARKS, i+1): None})
            data[pat_id] = pat_dict
        return data

