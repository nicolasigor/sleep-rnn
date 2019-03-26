from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import scipy.io

detector_path = '..'
sys.path.append(detector_path)

from sleep.mass import MASS
from sleep.inta import INTA
from sleep.data_ops import PATH_DATA

if __name__ == '__main__':
    dataset = MASS(load_checkpoint=True)
    scipy.io.savemat(os.path.join(PATH_DATA, 'mass.mat'),
                     mdict={'mass': dataset.data})
    dataset = INTA(load_checkpoint=True)
    scipy.io.savemat(os.path.join(PATH_DATA, 'inta.mat'),
                     mdict={'inta': dataset.data})
