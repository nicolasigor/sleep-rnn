"""Modules that defines operation to split training data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def simple_split_with_list(x, y, train_fraction=0.8, seed=None):
    """Splits data stored in a list.

    The data x and y are list of arrays with shape [batch, ...].
    These are split in two sets randomly using train_fraction over the number of
    element of the list. Then these sets are returned with
    the arrays concatenated along the first dimension
    """
    n_subjects = len(x)
    n_train = int(n_subjects * train_fraction)
    print('Split: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    x_train = np.concatenate([x[i] for i in train_idx], axis=0)
    y_train = np.concatenate([y[i] for i in train_idx], axis=0)
    x_test = np.concatenate([x[i] for i in test_idx], axis=0)
    y_test = np.concatenate([y[i] for i in test_idx], axis=0)
    return x_train, y_train, x_test, y_test


def split_ids_list(subject_ids, train_fraction=0.8, seed=None):
    """Splits the subject_ids list randomly using train_fraction."""
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_fraction)
    print('Split IDs: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    train_ids = [subject_ids[i] for i in train_idx]
    test_ids = [subject_ids[i] for i in test_idx]
    return train_ids, test_ids


def shuffle_data(x, y, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = x.shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    x = x[random_idx]
    y = y[random_idx]
    return x, y
