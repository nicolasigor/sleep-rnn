from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import pkeys, constants


if __name__ == '__main__':

    datasets_name_list = [
        # constants.NSRR_SS_NAME
        constants.CAP_SS_NAME,
        constants.PINK_NAME,
        constants.INTA_SS_NAME,
        constants.MODA_SS_NAME,
        constants.MASS_SS_NAME,
        constants.MASS_KC_NAME,
    ]
    repair_inta = False
    params = {pkeys.FS: 200}

    for dataset_name in datasets_name_list:
        # Create checkpoint and load to check
        print('')
        dataset = load_dataset(dataset_name, load_checkpoint=False, params=params, repair_inta=repair_inta)
        dataset.save_checkpoint()
        del dataset
        dataset = load_dataset(dataset_name, load_checkpoint=True, params=params)
        del dataset
