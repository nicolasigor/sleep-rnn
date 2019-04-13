from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np

detector_path = '..'
results_folder = 'results'
sys.path.append(detector_path)

from evaluation import metrics
from sleep.mass import MASS
from sleep.mass import KEY_MARKS
from sleep import data_ops


if __name__ == '__main__':

    dataset = MASS(load_checkpoint=True)

    subject_id = 1

    print('Preparing labels for testing... ', end='', flush=True)
    ind_dict = dataset.data[dataset.all_ids.index(subject_id)]
    marks_1 = ind_dict['%s_1' % KEY_MARKS]
    marks_2 = ind_dict['%s_2' % KEY_MARKS]
    marks_1 = data_ops.seq2inter(marks_1)
    marks_2 = data_ops.seq2inter(marks_2)
    print('Done')

    print('Matching events the old way... ', end='', flush=True)
    st = time.time()
    iou_array, idx_array = metrics.matching(marks_1, marks_2)
    print('Done (E.T. %1.6f [s])' % (time.time() - st), flush=True)

    print('Matching events the new way... ', end='', flush=True)
    st = time.time()
    iou_array_v2, idx_array_v2 = metrics.matching_v2(marks_1, marks_2)
    print('Done (E.T. %1.6f [s])' % (time.time() - st), flush=True)

    np.testing.assert_array_equal(iou_array, iou_array_v2)
    np.testing.assert_array_equal(idx_array, idx_array_v2)
    print('We are good!')
