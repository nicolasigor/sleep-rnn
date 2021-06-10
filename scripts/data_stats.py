from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.common import pkeys, constants


if __name__ == '__main__':
    dataset_name = constants.CAP_SS_NAME
    which_expert = 1

    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True
    dataset = reader.load_dataset(
        dataset_name, params=dataset_params, load_checkpoint=load_dataset_from_ckpt, verbose=False)

    stats = {}
    stats['n_subject'] = len(dataset.all_ids)
    stats['n_pages'] = np.concatenate(dataset.get_pages(pages_subset=constants.N2_RECORD)).size
    stats['hours'] = stats['n_pages'] * dataset.page_duration / 3600
    events = dataset.get_stamps(which_expert=which_expert, pages_subset=constants.N2_RECORD)
    stats['n_events'] = np.concatenate(events, axis=0).shape[0]
    if dataset_name == constants.MODA_SS_NAME:
        all_events = np.concatenate(events, axis=0)
        stats['densidad'] = stats['n_events'] / (stats['n_pages'] * dataset.page_duration / 60)
        stats['duration'] = np.mean(all_events[:, 1] - all_events[:, 0] + 1) / dataset.fs
    else:
        n_per_subject = [e.shape[0] for e in events]
        pages = dataset.get_pages(pages_subset=constants.N2_RECORD)
        min_per_subject = [(p.size * dataset.page_duration / 60) for p in pages]
        stats['densidad'] = np.mean([(number / minutes) for (number, minutes) in zip(n_per_subject, min_per_subject)])
        stats['duration'] = np.mean([np.mean(e[:, 1] - e[:, 0] + 1) for e in events]) / dataset.fs

    print('%s - expert %d' % (dataset_name, which_expert))
    for key in stats.keys():
        print(key, ':', stats[key])
