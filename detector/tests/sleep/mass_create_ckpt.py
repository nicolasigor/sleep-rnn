from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

detector_path = '../../'
sys.path.append(detector_path)

from sleep.mass import MASS

if __name__ == '__main__':
    dataset = MASS()
    dataset.save_checkpoint()
