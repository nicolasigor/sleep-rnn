from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

detector_path = '../..'
sys.path.append(detector_path)

from spectrum import cmorlet_v2
from spectrum import cmorlet

PATH_SAMPLE = '../demo_data/demo_signal.csv'

if __name__ == '__main__':
    fs = 200
    fb_list = [2.0]
    lower_freq = 2
    upper_freq = 30
    n_scales = 32

    demo_signal = np.loadtxt(PATH_SAMPLE)
    demo_signal = demo_signal[np.newaxis, :]
    signal_size = demo_signal.shape[1]
    print('Signal size:', signal_size)
    print('Signal shape:', demo_signal.shape)

    # Build graph
    signal_ph = tf.placeholder(tf.float32, shape=[None, signal_size])

    outputs_v1, np_wavelets_v1 = cmorlet.compute_cwt(
        signal_ph,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        flattening=True,
        border_crop=0,
        stride=1)

    outputs_v2, wavelets_v2 = cmorlet_v2.compute_cwt(
        signal_ph,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        size_factor=1.0,
        flattening=True,
        border_crop=0,
        stride=1,
        trainable=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np_outputs_v1, np_outputs_v2, np_wavelets_v2 = sess.run(
            [outputs_v1, outputs_v2, wavelets_v2],
            feed_dict={signal_ph: demo_signal}
        )

    # Just one spectrogram so
    np_wavelets_v1 = np_wavelets_v1[0]
    np_wavelets_v2 = np_wavelets_v2[0]

    # Let's divide in real and imag banks
    real_v1, imag_v1 = np_wavelets_v1
    real_v2, imag_v2 = np_wavelets_v2

    print('v1:', np_outputs_v1.shape, real_v1.shape, imag_v1.shape)
    print('v2:', np_outputs_v2.shape, real_v2.shape, imag_v2.shape)

    # Visualize a certain scale
    which_scale = -1

    real_scale_v1 = real_v1[0, :, 0, which_scale]
    imag_scale_v1 = imag_v1[0, :, 0, which_scale]
    real_scale_v2 = real_v2[0, :, 0, which_scale]
    imag_scale_v2 = imag_v2[0, :, 0, which_scale]

    time_axis_v1 = np.arange(real_scale_v1.size) / fs
    time_axis_v2 = np.arange(real_scale_v2.size) / fs

    # Show wavelets
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time_axis_v1, real_scale_v1, label='real')
    ax[0].plot(time_axis_v1, imag_scale_v1, label='imag')
    ax[0].set_title('V1')
    ax[0].legend()

    ax[1].plot(time_axis_v2, real_scale_v2, label='real')
    ax[1].plot(time_axis_v2, imag_scale_v2, label='imag')
    ax[1].set_title('V2')
    ax[1].legend()

    plt.show()
