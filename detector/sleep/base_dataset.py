"""Class definition to manipulate sleep spindle EEG datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from utils import constants
from utils import errors
from . import data_ops
from .data_ops import PATH_DATA

KEY_ID = 'subject_id'
KEY_EEG = 'signal'
KEY_PAGES = 'pages'
KEY_MARKS = 'marks'


class BaseDataset(object):
    """This is a base class sleep spindle datasets.
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
            n_experts,
            train_ids,
            test_ids,
    ):
        """Constructor.

        Args:
            dataset_dir: (String) Path to the folder containing the dataset.
               This path can be absolute, or relative to the project root.
            load_checkpoint: (Boolean). Whether to load from a checkpoint or to
               load from scratch using the original files of the dataset.
            name: (String) Name of the dataset. This name will be used for
               checkpoints.
            n_experts: (Int) Number of available sleep experts for sleep
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
        self._check_dataset_dir()  # We verify that the directory exists
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

        # Spindle and sleep EEG related parameters
        self.fs = 200  # Sampling frequency [Hz] to be used (not the original)
        self.page_duration = 20  # Time of window page [s]
        self.page_size = int(self.page_duration * self.fs)
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        # Data loading
        self.data = self._load_data()
        # n_pages = np.sum([ind[KEY_PAGES].shape[0] for ind in self.data])
        # print("\nPages in %s dataset: %s" % (self.name, n_pages))

    def get_subject_pages(self, subject_id):
        """Returns the indices of the N2 pages of this subject."""
        errors.check_valid_value(subject_id, 'ID', self.all_ids)
        # Look for dictionary associated with this id
        id_idx = self.all_ids.index(subject_id)
        ind_dict = self.data[id_idx]
        # Unpack data
        pages = ind_dict[KEY_PAGES]
        return pages

    def get_subject_data(
            self,
            subject_id,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            verbose=False
    ):
        """Returns segments of signal and marks from N2 pages for the given id.

        Args:
            subject_id: (int) id of the subject of interest.
            augmented_page: (Optional, boolean, defaults to False) whether to
                augment the page with half page at each side.
            border_size: (Optional, int, defaults to 0) number of samples to be
                added at each border of the segments.
            which_expert: (Optional, int, defaults to 1) Which expert
                annotations should be returned. It has to be consistent with
                the given n_experts, in a one-based counting.
            verbose: (Optional, boolean, defaults to False) Whether to print
                what is being read.

        Returns:
            n2_signal: (2D array) each row is an (augmented) page of the signal
            n2_marks: (2D array) each row is an (augmented) page of the marks
        """
        errors.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i+1) for i in range(self.n_experts)]
        errors.check_valid_value(which_expert, 'which_expert', valid_experts)

        # Look for dictionary associated with this id
        id_idx = self.all_ids.index(subject_id)
        ind_dict = self.data[id_idx]
        # Unpack data
        signal = ind_dict[KEY_EEG]
        n2_pages = ind_dict[KEY_PAGES]
        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
        # Compute border to be added
        if augmented_page:
            total_border = self.page_size // 2 + border_size
        else:
            total_border = border_size
        n2_signal = data_ops.extract_pages(
            signal, n2_pages, self.page_size, border_size=total_border)
        n2_marks = data_ops.extract_pages(
            marks, n2_pages, self.page_size, border_size=total_border)

        if verbose:
            print('Getting ID %s, %d N2 pages, Expert %d'
                  % (ind_dict[KEY_ID], n2_pages.size, which_expert))
        return n2_signal, n2_marks

    def get_subset(
            self,
            subject_id_list,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            verbose=False
    ):
        """Returns the list of signals and marks from a list of subjects.
        """
        subset_signals = []
        subset_marks = []
        for subject_id in subject_id_list:
            n2_signal, n2_marks = self.get_subject_data(
                subject_id,
                augmented_page=augmented_page,
                border_size=border_size,
                which_expert=which_expert,
                verbose=verbose)
            subset_signals.append(n2_signal)
            subset_marks.append(n2_marks)
        return subset_signals, subset_marks

    def get_train_subset(
            self,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            verbose=False
    ):
        """Returns the train subset"""
        train_signals, train_marks = self.get_subset(
            self.train_ids,
            augmented_page=augmented_page,
            border_size=border_size,
            which_expert=which_expert,
            verbose=verbose)
        return train_signals, train_marks

    def get_test_subset(
            self,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            verbose=False
    ):
        """Returns the test subset"""
        test_signals, test_marks = self.get_subset(
            self.test_ids,
            augmented_page=augmented_page,
            border_size=border_size,
            which_expert=which_expert,
            verbose=verbose)
        return test_signals, test_marks

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
            print('Loading from checkpoint')
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

    def _check_dataset_dir(self):
        """Checks if the directory containing the data exists"""
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(
                'Directory not found: %s' % self.dataset_dir)

    def _load_from_files(self):
        """Loads and return the data from files and transforms it appropriately.
        This is just a template for the specific implementation of the dataset.
        the value of the key KEY_ID has to be an integer.
        """
        # List of dictionaries containing the data
        data = []
        for pat_id in self.all_ids:
            pat_dict = {KEY_ID: pat_id, KEY_EEG: None, KEY_PAGES: None}
            for i in range(self.n_experts):
                pat_dict.update({'%s_%d' % (KEY_MARKS, i+1): None})
            data.append(pat_dict)
        return data
