from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.dreams_kc import DreamsKC
from sleeprnn.data.dreams_ss import DreamsSS
from sleeprnn.data.mass_kc import MassKC
from sleeprnn.data.mass_ss import MassSS
from sleeprnn.data.inta_ss import IntaSS


if __name__ == '__main__':

    datasets_class = [DreamsSS, DreamsKC, MassSS, MassKC, IntaSS]
    repair_inta = False

    for data_class in datasets_class:
        # Create checkpoint and load to check
        print('')
        if data_class == IntaSS:
            dataset = data_class(repair_stamps=repair_inta)
        else:
            dataset = data_class()
        dataset.save_checkpoint()
        del dataset
        dataset = data_class(load_checkpoint=True)
        del dataset
