from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

detector_path = '..'
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA

if __name__ == '__main__':
    # MASS - Create checkpoint and load to check
    dataset = MASS()
    dataset.save_checkpoint()
    del dataset
    dataset = MASS(load_checkpoint=True)
    del dataset

    # INTA - Create checkpoint and load to check
    dataset = INTA()
    dataset.save_checkpoint()
    del dataset
    dataset = INTA(load_checkpoint=True)
