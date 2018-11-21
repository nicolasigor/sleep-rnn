from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

detector_path = '../../'
sys.path.append(detector_path)

from neuralnet.networks import wavelet_blstm_net
from utils import constants
from utils import param_keys


if __name__ == '__main__':
    # Parameters
    params = param_keys.default_params
    page_size = params[param_keys.PAGE_SIZE]
    border_size = params[param_keys.BORDER_CROP]
    input_length = int(page_size + 2*border_size)

    # Build computational
    tf.reset_default_graph()
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_length],
                            name="feats_train_ph")

    outputs = wavelet_blstm_net(
        inputs,
        params,
        training=True,
        name='model')

    sess = tf.Session()
    tb_path = os.path.join(os.path.dirname(__file__), 'logs')
    writer = tf.summary.FileWriter(os.path.join(tb_path, 'train'), sess.graph)
    print('Saving summaries at %s' % tb_path)
