from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.dataset import Dataset
from sleeprnn.data import utils
from sleeprnn.helpers import reader


def get_partitions(dataset: Dataset, strategy, n_seeds):
    train_ids_list = []
    val_ids_list = []
    test_ids_list = []
    if strategy == 'fixed':
        for fold_id in range(n_seeds):
            train_ids, val_ids = utils.split_ids_list_v2(dataset.train_ids, split_id=fold_id)
            train_ids_list.append(train_ids)
            val_ids_list.append(val_ids)
            test_ids_list.append(dataset.test_ids)
    elif strategy == '5cv':
        for cv_seed in range(n_seeds):
            for fold_id in range(5):
                train_ids, val_ids, test_ids = dataset.cv_split(5, fold_id, cv_seed)
                train_ids_list.append(train_ids)
                val_ids_list.append(val_ids)
                test_ids_list.append(test_ids)
    else:
        raise ValueError
    return train_ids_list, val_ids_list, test_ids_list


if __name__ == '__main__':
    dataset_name = "moda_ss"
    strategy = "5cv"
    n_seeds = 3

    dataset_name_to_save = dataset_name.split("_")[0]
    dataset = reader.load_dataset(dataset_name)
    train_ids_list, val_ids_list, test_ids_list = get_partitions(dataset, strategy, n_seeds)
    partition_id = "%s_%s" % (dataset_name_to_save, strategy)
    partition_dict = {
        'dataset_name': dataset_name_to_save,
        'strategy': strategy,
        'n_seeds': n_seeds,
        'train': train_ids_list,
        'val': val_ids_list,
        'test': test_ids_list
    }
    save_dir = os.path.abspath("../resources/datasets/exported_partitions")
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, 'partitions_%s.json' % partition_id)
    with open(fname, 'w') as outfile:
        json.dump(partition_dict, outfile)
