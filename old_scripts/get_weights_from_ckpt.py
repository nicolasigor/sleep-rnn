from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from tensorflow.python import pywrap_tensorflow
import pickle

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

project_root = os.path.abspath('..')
sys.path.append(project_root)

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    ckpt_path = os.path.join(RESULTS_PATH, 'ckpt_v19_mass_ss_e1/seed0/model/ckpt')
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    name_shape_dict = reader.get_variable_to_shape_map()
    names_list = list(reader.get_variable_to_shape_map().keys())
    names_list.sort()

    bn_cwt_weights = {}
    for tensor_name in names_list:
        if len(name_shape_dict[tensor_name]) == 0 or 'Adam' in tensor_name or 'opaque_kernel' in tensor_name:
            continue
        splited_name = tensor_name.split('/')
        if 'spectrum' in splited_name and 'batch_normalization' in splited_name:
            tmp_dict = {
                splited_name[-1]: reader.get_tensor(tensor_name)
            }
            bn_cwt_weights.update(tmp_dict)

    print(bn_cwt_weights)
    for key in bn_cwt_weights:
        print('Tensor %s with shape' % key, bn_cwt_weights[key].shape)
    np.savez(
        'bn_cwt_weights.npz', **bn_cwt_weights)
