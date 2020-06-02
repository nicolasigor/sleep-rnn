from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.mass_kc import MassKC
from sleeprnn.data.mass_ss import MassSS
from sleeprnn.data.inta_ss import IntaSS
from sleeprnn.common import pkeys


if __name__ == '__main__':

    datasets_class = [MassSS, MassKC, IntaSS]
    repair_inta = False
    params = {pkeys.FS: 200}

    for data_class in datasets_class:
        # Create checkpoint and load to check
        print('')
        if data_class == IntaSS:
            dataset = data_class(repair_stamps=repair_inta, params=params)
        else:
            dataset = data_class(params=params)
        dataset.save_checkpoint()
        del dataset
        dataset = data_class(params=params, load_checkpoint=True)
        del dataset
