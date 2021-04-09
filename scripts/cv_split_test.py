import os
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('..')

from sleeprnn.common import viz, constants, pkeys
from sleeprnn.data import utils
from sleeprnn.helpers import reader


if __name__ == "__main__":
    dataset = reader.load_dataset(constants.MODA_SS_NAME)

    n_folds = 5
    seed = 1

    print("\nChecking %d folds and seed %s" % (n_folds, seed))

    # Check overlap
    for fold_id in range(n_folds):
        train_ids, val_ids, test_ids = dataset.cv_split(n_folds, fold_id, seed=seed)
        print("fold %d, train %d, val %d, test %d, total unique %d" % (
            fold_id, train_ids.size, val_ids.size, test_ids.size,
            np.unique(np.concatenate([train_ids, val_ids, test_ids])).size
        ))
    # Check continuity of test and val folds
    test_folds = []
    val_folds = []
    for fold_id in range(n_folds):
        _, val_ids, test_ids = dataset.cv_split(n_folds, fold_id, seed=seed)
        test_folds.append(test_ids)
        val_folds.append(val_ids)
    test_folds_shifted = test_folds[1:] + test_folds[:1]
    if np.all(np.concatenate(val_folds) == np.concatenate(test_folds_shifted)):
        print("Val folds correct")
    else:
        print("Val folds incorrect")
    # Check independence of test sets of different folds
    print("Total unique test sets %d" % np.unique(np.concatenate(test_folds)).size)
    print("Total unique val sets %d" % np.unique(np.concatenate(val_folds)).size)
    # Check stratification
    print("\nCheck stratified CV")
    for fold_id in range(n_folds):
        train_ids, val_ids, test_ids = dataset.cv_split(n_folds, fold_id, seed=seed)
        print("\nfold %d" % fold_id)
        ids_dict = {
            'train': train_ids,
            'val  ': val_ids,
            'test ': test_ids
        }
        for set_name in ids_dict.keys():
            set_ids = ids_dict[set_name]
            phases = np.asarray([dataset.data[s]['phase'] for s in set_ids])
            n_blocks = np.asarray([dataset.data[s]['n_blocks'] for s in set_ids])
            print("%s, p1 %d, p2 %d, n10 %d, n3 %d, p1_n10 %d, p2_n10 %d, p1_n3 %d, p2_n3 %d" % (
                set_name,
                np.sum(phases == 1),
                np.sum(phases == 2),
                np.sum(n_blocks == 10),
                np.sum(n_blocks < 10),
                np.sum((phases == 1) & (n_blocks == 10)),
                np.sum((phases == 2) & (n_blocks == 10)),
                np.sum((phases == 1) & (n_blocks < 10)),
                np.sum((phases == 2) & (n_blocks < 10)),
            ))

    # Check sorted
    for fold_id in range(n_folds):
        train_ids, val_ids, test_ids = dataset.cv_split(n_folds, fold_id, seed=seed)
        print("\nfold %d" % fold_id)
        print("train", train_ids)
        print("val", val_ids)
        print("test", test_ids)
