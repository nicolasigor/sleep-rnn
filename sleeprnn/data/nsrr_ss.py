from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle

import numpy as np
import pandas as pd

from sleeprnn.common import constants
from sleeprnn.data import utils
from sleeprnn.data import stamp_correction
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS, KEY_HYPNOGRAM

PATH_NSRR_RELATIVE = 'nsrr'
PATH_REC_AND_STATE = 'register_and_state'
SUBDATASETS = ['shhs1', 'mros1', 'chat1', 'sof', 'cfs', 'ccshs']


class NsrrSS(Dataset):
    """This is a class to manipulate the NSRR data EEG dataset.
    """

    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        """Constructor"""
        # NSRR parameters
        self.state_ids = np.array([
            'Wake|0',
            'Stage 1 sleep|1',
            'Stage 2 sleep|2',
            'Stage 3 sleep|3',
            'Stage 4 sleep|4',
            'REM sleep|5',
            'Movement|6',
            'Unscored|9'])
        self.unknown_id = 'Unscored|9'  # Character for unknown state in hypnogram
        self.n2_id = 'Stage 2 sleep|2'  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        all_ids = [1, 2, 3]  # Dummy, will be established after processing data
        all_ids.sort()

        hypnogram_sleep_labels = np.array([
            'Stage 1 sleep|1',
            'Stage 2 sleep|2',
            'Stage 3 sleep|3',
            'Stage 4 sleep|4',
            'REM sleep|5'])
        hypnogram_page_duration = 30

        super(NsrrSS, self).__init__(
            dataset_dir=PATH_NSRR_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.NSRR_SS_NAME,
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            hypnogram_sleep_labels=hypnogram_sleep_labels,
            hypnogram_page_duration=hypnogram_page_duration,
            n_experts=1,  # Dummy
            params=params,
            verbose=verbose
        )
        self.global_std = None
        if verbose:
            print('Global STD:', self.global_std)
        if verbose:
            print('Dataset %s with %d patients.' % (self.dataset_name, len(self.all_ids)))

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        save_dir = os.path.join(self.dataset_dir, 'pretty_files')
        os.makedirs(save_dir, exist_ok=True)
        start = time.time()
        for i_dataset, subdataset in enumerate(data_paths.keys()):
            print("\nLoading subdataset %s" % subdataset)
            meta_df = pd.read_csv(data_paths[subdataset]['metadata'])
            meta_dict = meta_df.set_index('subject_id').to_dict(orient='index')

            subject_paths = data_paths[subdataset]['eeg_and_state']
            subject_ids = list(subject_paths.keys())
            for i_subject, subject_id in enumerate(subject_ids[:5]):
                print('\nLoading ID %s' % subject_id)

                subject_eeg_state_file = subject_paths[subject_id]
                subject_meta = meta_dict[subject_id]
                print(subject_eeg_state_file)
                print(subject_meta)


        #for i, subject_id in enumerate(data_paths.keys()):
            # print('\nLoading ID %s' % subject_id)
            # path_dict = data_paths[subject_id]
            # # Read data
            # signal, hypnogram_original = self._read_npz(path_dict[KEY_FILE_EEG_STATE])
            # n2_pages = self._get_n2_pages(hypnogram_original)
            # marks_1 = self._read_marks(path_dict['%s_1' % KEY_FILE_MARKS])
            # signal, n2_pages, marks_1, hypnogram_20s = self._short_signals(signal, n2_pages, marks_1)
            #
            # # ################
            # # Remove weird pages from N2 labels
            # weird_locs = np.where(np.abs(signal) > 300)[0]
            # weird_pages = np.unique(np.floor(weird_locs / self.page_size)).astype(np.int32)
            # if subject_id == '2-001':
            #     weird_pages = np.concatenate([
            #         weird_pages,
            #         np.arange(0, 60 + 0.001),
            #         np.arange(120, 130 + 0.001)
            #     ])
            #     weird_pages = np.unique(weird_pages).astype(np.int32)
            # hypnogram_20s[weird_pages] = self.unknown_id
            # n2_pages = np.where(hypnogram_20s == self.n2_id)[0]
            # # ################
            #
            # total_pages = int(signal.size / self.page_size)
            # all_pages = np.arange(1, total_pages - 1, dtype=np.int16)
            #
            # # Marks from simple detectors S1 (abs) and S2 (rel), respectively
            # marks_2 = self._read_marks_simple(path_dict['%s_2' % KEY_FILE_MARKS])
            # marks_3 = self._read_marks_simple(path_dict['%s_3' % KEY_FILE_MARKS])
            #
            # print('N2 pages: %d' % n2_pages.shape[0])
            # print('Whole-night pages: %d' % all_pages.shape[0])
            # print('Marks SS from A7 with original paper params         : %d' % marks_1.shape[0])
            # print('Marks SS from S1-abs with thr 10 uV and thr low 0.86: %d' % marks_2.shape[0])
            # print('Marks SS from S2-rel with thr 2.9 and thr low 0.80  : %d' % marks_3.shape[0])
            #
            # # Save data
            # ind_dict = {
            #     KEY_EEG: signal.astype(np.float32),
            #     KEY_N2_PAGES: n2_pages.astype(np.int16),
            #     KEY_ALL_PAGES: all_pages.astype(np.int16),
            #     KEY_HYPNOGRAM: hypnogram_20s,
            #     '%s_1' % KEY_MARKS: marks_1.astype(np.int32),
            #     '%s_2' % KEY_MARKS: marks_2.astype(np.int32),
            #     '%s_3' % KEY_MARKS: marks_3.astype(np.int32),
            # }
            # ind_dict = {}
            #
            # # Save data to disk and only save in object the path
            # fname = os.path.join(save_dir, 'subject_%s.npz' % subject_id)
            # np.savez(fname, **ind_dict)
            #
            # data[subject_id] = {'pretty_file_path': fname}
            #
            # print('Loaded ID %s (%02d/%02d ready). Time elapsed: %1.4f [s]' % (
            #     subject_id, i+1, n_data, time.time()-start))
        # print('%d records have been read.' % n_data)
        return data

    def read_subject_data(self, subject_id):
        path_dict = self.data[subject_id]
        ind_dict = np.load(path_dict['pretty_file_path'])

        loaded_ind_dict = {}
        for key in ind_dict.files:
            loaded_ind_dict[key] = ind_dict[key]

        return loaded_ind_dict

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        all_ids = []
        for subdataset in SUBDATASETS:
            data_dir = os.path.join(self.dataset_dir, subdataset)
            eeg_dir = os.path.join(data_dir, PATH_REC_AND_STATE)
            meta_file = [f for f in os.listdir(data_dir) if 'metadata.csv' in f][0]
            meta_file = os.path.join(data_dir, meta_file)
            subject_ids = np.array(
                [".".join(f.split(".")[:-1]) for f in os.listdir(eeg_dir) if 'npz' in f], dtype='<U40')
            # Only keep those subject ids that intersect with metafile subject ids
            subject_ids_meta = pd.read_csv(meta_file)['subject_id'].values
            subject_ids_common = list(set.intersection(set(subject_ids), set(subject_ids_meta)))
            subdataset_paths = {
                'metadata': meta_file,
                'eeg_and_state': {
                    s: os.path.join(eeg_dir, '%s.npz' % s)
                    for s in subject_ids_common}
            }
            data_paths[subdataset] = subdataset_paths
            # Collect IDs
            all_ids.append(np.array(subject_ids_common, dtype='<U40'))
        all_ids = np.sort(np.concatenate(all_ids))
        # Replace all_ids dummy
        self.all_ids = all_ids

        print('%d records in %s dataset.' % (len(self.all_ids), self.dataset_name))
        return data_paths

    def _load_from_checkpoint(self):
        """Loads the pickle file containing the loaded data."""
        with open(self.ckpt_file, 'rb') as handle:
            data = pickle.load(handle)
        all_ids = list(data.keys())
        all_ids = np.sort(np.array(all_ids, dtype='<U40'))
        # Replace all_ids dummy
        self.all_ids = all_ids
        return data

