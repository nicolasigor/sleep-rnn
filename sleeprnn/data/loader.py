from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

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


def replace_submodule_in_module(module, old_submodule, new_submodule):
    module_splitted = module.split(".")
    if old_submodule in module_splitted:
        idx_name = module_splitted.index(old_submodule)
        module_splitted[idx_name] = new_submodule
    new_module = ".".join(module_splitted)
    return new_module


class RefactorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = replace_submodule_in_module(module, 'sleep', 'sleeprnn')
        module = replace_submodule_in_module(module, 'neuralnet', 'nn')
        return super().find_class(module, name)
