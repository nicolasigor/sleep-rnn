from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

detector_path = '../../'
sys.path.append(detector_path)

from neuralnet.networks import cmorlet_conv_blstm_net
from utils import constants


if __name__ == '__main__':
    # Parameters
    fs = 200
    page_duration = 20
    border_duration = 1
    fb_list = [1.5]
    border_crop = int(border_duration * fs)
    training = True
    input_length = int(fs*(page_duration + 2*border_duration))

    # Build computational
    tf.reset_default_graph()
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_length],
                            name="feats_train_ph")
    outputs = cmorlet_conv_blstm_net(
        inputs,
        fb_list,
        fs,
        border_crop,
        training,
        n_conv_blocks=2,
        n_time_levels=3,
        batchnorm_conv=constants.BN_RENORM,
        batchnorm_first_lstm=constants.BN_RENORM,
        dropout_first_lstm=None,
        batchnorm_rest_lstm=None,
        dropout_rest_lstm=constants.SEQUENCE_DROP,
        time_pooling=constants.AVGPOOL,
        batchnorm_fc=None,
        dropout_fc=constants.SEQUENCE_DROP,
        drop_rate=0.5,
        trainable_wavelet=False,
        name='model')

    sess = tf.Session()
    tb_path = os.path.join(os.path.dirname(__file__), 'logs')
    writer = tf.summary.FileWriter(os.path.join(tb_path, 'train'), sess.graph)
    print('Saving summaries at %s' % tb_path)
