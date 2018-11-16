from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

detector_path = '../../'
sys.path.append(detector_path)

from sleep.inta import INTA

if __name__ == '__main__':
    dataset = INTA()
    dataset.save_checkpoint()
    del dataset
    dataset = INTA(load_checkpoint=True)
