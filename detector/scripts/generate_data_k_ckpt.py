from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

detector_path = '..'
sys.path.append(detector_path)

from sleep.mass_k import MASSK


if __name__ == '__main__':
    # MASS - Create checkpoint and load to check
    dataset = MASSK()
    dataset.save_checkpoint()
    del dataset
    dataset = MASSK(load_checkpoint=True)
    del dataset
