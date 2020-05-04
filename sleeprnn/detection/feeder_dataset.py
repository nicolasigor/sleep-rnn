"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_MARKS, KEY_N2_PAGES, KEY_ALL_PAGES
from sleeprnn.data import utils
from sleeprnn.common import constants, checks


class FeederDataset(Dataset):

    def __init__(
            self,
            dataset: Dataset,
            sub_ids,
            task_mode,
            which_expert=1,
            verbose=False
    ):

        """Constructor"""
        checks.check_valid_value(
            task_mode, 'task_mode',
            [constants.WN_RECORD, constants.N2_RECORD])

        self.parent_dataset = dataset
        self.task_mode = task_mode
        self.which_expert = which_expert

        super(FeederDataset, self).__init__(
            dataset_dir=dataset.dataset_dir,
            load_checkpoint=False,
            dataset_name='%s_subset' % dataset.dataset_name,
            all_ids=sub_ids,
            event_name=dataset.event_name,
            n_experts=dataset.n_experts,
            params=dataset.params.copy(),
            verbose=verbose
        )
        self.global_std = dataset.global_std

    def _load_from_source(self):
        """Loads the data from source."""
        data = self.parent_dataset.get_sub_dataset(self.all_ids)
        self.parent_dataset = None
        return data

    def get_stamps(
            self,
            which_expert=None,
            pages_subset=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        if pages_subset is None:
            pages_subset = self.task_mode
        subset_marks = super(FeederDataset, self).get_stamps(
            which_expert=which_expert,
            pages_subset=pages_subset,
            verbose=verbose)
        return subset_marks

    def get_subset_stamps(
            self,
            subject_id_list,
            which_expert=None,
            pages_subset=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        if pages_subset is None:
            pages_subset = self.task_mode
        subset_marks = super(FeederDataset, self).get_subset_stamps(
            subject_id_list=subject_id_list,
            which_expert=which_expert,
            pages_subset=pages_subset,
            verbose=verbose)
        return subset_marks

    def get_subject_stamps(
            self,
            subject_id,
            which_expert=None,
            pages_subset=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        if pages_subset is None:
            pages_subset = self.task_mode
        marks = super(FeederDataset, self).get_subject_stamps(
            subject_id=subject_id,
            which_expert=which_expert,
            pages_subset=pages_subset,
            verbose=verbose)
        return marks

    def get_subject_signal(
            self,
            subject_id,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        signal = super(FeederDataset, self).get_subject_signal(
            subject_id=subject_id,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            which_expert=which_expert,
            verbose=verbose
        )
        return signal

    def get_subset_signals(
            self,
            subject_id_list,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        subset_signals = super(FeederDataset, self).get_subset_signals(
            subject_id_list=subject_id_list,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            which_expert=which_expert,
            verbose=verbose
        )
        return subset_signals

    def get_signals(
            self,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        subset_signals = super(FeederDataset, self).get_signals(
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            which_expert=which_expert,
            verbose=verbose
        )
        return subset_signals

    def get_subject_data(
            self,
            subject_id,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=None,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        signal, marks = super(FeederDataset, self).get_subject_data(
            subject_id=subject_id,
            augmented_page=augmented_page,
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            which_expert=which_expert,
            pages_subset=pages_subset,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            verbose=verbose
        )
        return signal, marks

    def get_subset_data(
            self,
            subject_id_list,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=None,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        subset_signals, subset_marks = super(FeederDataset, self).get_subset_data(
            subject_id_list=subject_id_list,
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

    def get_data(
            self,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=None,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.which_expert
        subset_signals, subset_marks = super(FeederDataset, self).get_data(
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

    def get_data_for_training(
            self,
            border_size=0,
            forced_mark_separation_size=0,
            verbose=False
    ):
        signals, stamps = super().get_data(
            augmented_page=True,
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            which_expert=self.which_expert,
            pages_subset=self.task_mode,
            normalize_clip=True,
            normalization_mode=self.task_mode,
            verbose=verbose)
        return signals, stamps

    def get_data_for_prediction(
            self,
            border_size=0,
            predict_with_augmented_page=True,
            verbose=False
    ):
        signals, stamps = super().get_data(
            augmented_page=predict_with_augmented_page,
            border_size=border_size,
            which_expert=self.which_expert,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=self.task_mode,
            verbose=verbose)
        return signals, stamps

    def get_data_for_stats(
            self,
            border_size=0,
            verbose=False
    ):
        subset_signals = []
        for subject_id in self.all_ids:
            signal = self.get_subject_data_for_stats(
                subject_id=subject_id,
                border_size=border_size,
                verbose=verbose)
            subset_signals.append(signal)
        return subset_signals

    def get_subject_data_for_stats(
            self,
            subject_id,
            border_size=0,
            verbose=False
    ):
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        ind_dict = self.data[subject_id]

        # Unpack data
        signal = ind_dict[KEY_EEG]
        marks = ind_dict['%s_%d' % (KEY_MARKS, self.which_expert)]
        # Transform stamps into sequence
        marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)

        if self.task_mode == constants.WN_RECORD:
            if verbose:
                print('Stats from pages containing true events.')
            # Normalize using stats from pages with true events.
            stat_pages = ind_dict[KEY_ALL_PAGES]
            activity = utils.extract_pages(
                marks, stat_pages,
                self.page_size, border_size=0)
            activity = activity.sum(axis=1)
            activity = np.where(activity > 0)[0]
            stat_pages = stat_pages[activity]
        else:
            if verbose:
                print('Stats from N2 pages.')
            stat_pages = ind_dict[KEY_N2_PAGES]
        signal, _ = utils.norm_clip_signal(
            signal, stat_pages, self.page_size)
        # Extract segments
        signal = utils.extract_pages(
            signal, stat_pages, self.page_size, border_size=border_size)
        return signal
