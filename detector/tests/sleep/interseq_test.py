from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

detector_path = '../../'
sys.path.append(detector_path)

from sleep.data_ops import inter2seq, seq2inter, seq2inter_with_pages


if __name__ == '__main__':
    example_seq = np.array(
        [0, 0, 1, 1, 0, 1, 0, 0]
    )
    example_seq2inter = np.array(
        [
            [2, 3],
            [5, 5]
        ]
    )

    print('Example sequence:\n', example_seq)
    print('True conversion to intervals:\n', example_seq2inter)
    print('Conversion to intervals:\n', seq2inter(example_seq))

    example_inter = np.array(
        [
            [1, 3],
            [6, 8],
            [10, 10]
        ]
    )
    example_inter2seq = np.array(
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    )

    print('Example intervals:\n', example_inter)
    print('True conversion to sequence:\n', example_inter2seq)
    print('Conversion to sequence:\n', inter2seq(example_inter, 0, 10))

    example_seq_pages = np.array([
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 1]
    ])
    example_pages = np.array(
        [1, 3, 5, 6]
    )
    example_global_seq = np.array(
        [0, 0, 0, 0,
         0, 0, 1, 1,
         0, 0, 0, 0,
         1, 1, 0, 0,
         0, 0, 0, 0,
         0, 0, 1, 1,
         1, 1, 0, 1]
    )
    example_inter_pages = seq2inter(example_global_seq)

    print('Example sequence with pages:\n', example_seq_pages)
    print('Provided pages for example:\n', example_pages)
    print('True conversion to intervals:\n', example_inter_pages)
    print('Conversion to intervals:\n', seq2inter_with_pages(example_seq_pages, example_pages))
