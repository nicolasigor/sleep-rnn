from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from sleep.data.mass_ss import MassSS
from sleep.data.mass_kc import MassKC
from sleep.data.inta_ss import IntaSS


if __name__ == '__main__':

    datasets_class = [MassSS, MassKC, IntaSS]

    for data_class in datasets_class:
        # Create checkpoint and load to check
        print('')
        dataset = data_class()
        dataset.save_checkpoint()
        del dataset
        dataset = data_class(load_checkpoint=True)
        del dataset
