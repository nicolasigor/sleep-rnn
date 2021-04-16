from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.mass_kc import MassKC
from sleeprnn.data.mass_ss import MassSS
from sleeprnn.data.mass_ss_alt import MassSSAlt
from sleeprnn.data.moda_ss import ModaSS
from sleeprnn.data.inta_ss import IntaSS
from sleeprnn.data.mass_kc_inv import MassKCInv
from sleeprnn.data.mass_ss_inv import MassSSInv
from sleeprnn.data.cap_full_ss import CapFullSS
from sleeprnn.common import pkeys


if __name__ == '__main__':

    datasets_class = [IntaSS]  # [ModaSS]  # [MassSS, MassKC, MassSSInv, MassKCInv]
    repair_inta = False
    params = {pkeys.FS: 200}
    mass_global_std = None

    for data_class in datasets_class:
        # Create checkpoint and load to check
        print('')

        if data_class == IntaSS:
            dataset = data_class(repair_stamps=repair_inta, params=params)
        elif data_class in [MassSSInv, MassKCInv]:
            if mass_global_std is None:
                ref_dataset = MassSS(params=params, load_checkpoint=True)
                mass_global_std = ref_dataset.global_std
                del ref_dataset
                print("Computed global_std", mass_global_std)
            dataset = data_class(mass_global_std, params=params)
        else:
            dataset = data_class(params=params)

        if data_class in [MassSS, MassKC] and mass_global_std is None:
            mass_global_std = dataset.global_std
            print("Saved MASS global std", mass_global_std)

        dataset.save_checkpoint()
        del dataset

        if data_class in [MassSSInv, MassKCInv]:
            dataset = data_class(mass_global_std, params=params, load_checkpoint=True)
        else:
            dataset = data_class(params=params, load_checkpoint=True)

        del dataset
