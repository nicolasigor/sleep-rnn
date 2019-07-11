from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sleeprnn.common import checks, constants
from .dreams_kc import DreamsKC
from .dreams_ss import DreamsSS
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
            constants.INTA_SS_NAME,
            constants.DREAMS_KC_NAME,
            constants.DREAMS_SS_NAME
        ])
    if dataset_name == constants.MASS_SS_NAME:
        dataset = MassSS(load_checkpoint=load_checkpoint, params=params)
    elif dataset_name == constants.MASS_KC_NAME:
        dataset = MassKC(load_checkpoint=load_checkpoint, params=params)
    elif dataset_name == constants.INTA_SS_NAME:
        dataset = IntaSS(load_checkpoint=load_checkpoint, params=params)
    elif dataset_name == constants.DREAMS_SS_NAME:
        dataset = DreamsSS(load_checkpoint=load_checkpoint, params=params)
    else:
        dataset = DreamsKC(load_checkpoint=load_checkpoint, params=params)
    return dataset
