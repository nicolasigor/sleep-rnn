from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sleeprnn.common import checks, constants
from .inta_ss import IntaSS
from .mass_kc import MassKC
from .mass_ss import MassSS


def load_dataset(dataset_name, load_checkpoint=True, params=None):
    # Load data
    checks.check_valid_value(
        dataset_name, 'dataset_name',
        [
            constants.MASS_KC_NAME,
            constants.MASS_SS_NAME,
            constants.INTA_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=load_checkpoint, params=params)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(load_checkpoint=load_checkpoint, params=params)
    else:
        dataset = IntaSS(load_checkpoint=load_checkpoint, params=params)
    return dataset
