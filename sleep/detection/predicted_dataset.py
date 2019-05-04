"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sleep.data.dataset import Dataset, KEY_MARKS
from sleep.common import constants
from .feeder_dataset import FeederDataset
from .postprocessor import PostProcessor


class PredictedDataset(Dataset):

    def __init__(
            self,
            dataset: FeederDataset,
            probabilities_dict,
            params=None
    ):
        self.parent_dataset = dataset
        self.task_mode = dataset.task_mode
        self.probabilities_dict = probabilities_dict
        self.postprocessor = PostProcessor(
            event_name=dataset.event_name, params=params)
        self.probability_threshold = None

        """Constructor"""
        super().__init__(
            dataset_dir=dataset.dataset_dir,
            load_checkpoint=False,
            dataset_name='%s_predicted' % dataset.dataset_name,
            all_ids=dataset.all_ids,
            event_name=dataset.event_name,
            n_experts=1,
            params=dataset.params
        )

        # Check that subject ids in probabilities are the same as the ones
        # on the dataset
        ids_proba = list(self.probabilities_dict.keys())
        ids_data = dataset.all_ids
        ids_proba.sort()
        if ids_data != ids_proba:
            raise ValueError(
                'IDs mismatch: IDs from predictions are %s '
                'but IDs from given dataset are %s' % (ids_proba, ids_data))

        self.set_probability_threshold(0.5)

    def _load_from_source(self):
        """Loads the data from source."""
        data = self.parent_dataset.get_sub_dataset(self.all_ids)
        return data

    def set_probability_threshold(self, new_probability_threshold):
        self.probability_threshold = new_probability_threshold
        self._update_stamps()

    def _update_stamps(self):

        print('Producing probability list')
        probabilities_list = []
        for sub_id in self.all_ids:
            probabilities_list.append(self.probabilities_dict[sub_id])
        print('Probability list size', len(probabilities_list))

        wn_pages_val = self.get_pages(pages_subset=constants.WN_RECORD)
        if self.task_mode == constants.N2_RECORD:
            # Keep only N2 stamps
            n2_pages_val = self.get_pages(
                pages_subset=constants.N2_RECORD)
        else:
            n2_pages_val = None
        print('WN pages list size', len(wn_pages_val))

        print('Thr', self.probability_threshold)
        stamps_list = self.postprocessor.proba2stamps_with_list(
            probabilities_list,
            wn_pages_val,
            pages_indices_subset=n2_pages_val,
            thr=self.probability_threshold)
        print('Size stamp list', len(stamps_list))
        print('First stamp', stamps_list[0].shape)
        # Now save model stamps
        stamp_key = '%s_%d' % (KEY_MARKS, 1)

        print('Old first stamps (expert)', self.data[self.all_ids[0]][stamp_key].shape)


        for k, sub_id in enumerate(self.all_ids):
            print('Keys Id ', sub_id)
            print(self.data[sub_id].keys())
            self.data[sub_id][stamp_key] = stamps_list[k]
        print('New first stamps (expert)', self.data[self.all_ids[0]][stamp_key].shape)