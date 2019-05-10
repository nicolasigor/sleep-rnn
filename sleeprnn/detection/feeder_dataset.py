"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sleeprnn.data.dataset import Dataset
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

        super().__init__(
            dataset_dir=dataset.dataset_dir,
            load_checkpoint=False,
            dataset_name='%s_subset' % dataset.dataset_name,
            all_ids=sub_ids,
            event_name=dataset.event_name,
            n_experts=dataset.n_experts,
            params=dataset.params,
            verbose=verbose
        )

    def _load_from_source(self):
        """Loads the data from source."""
        data = self.parent_dataset.get_sub_dataset(self.all_ids)
        self.parent_dataset = None
        return data

    def get_data_for_training(
            self,
            border_size=0,
            verbose=False
    ):
        signals, stamps = super().get_data(
            augmented_page=True,
            border_size=border_size,
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
