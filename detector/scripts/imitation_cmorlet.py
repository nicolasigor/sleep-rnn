from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

detector_path = '..'
sys.path.append(detector_path)

from spectrum import cmorlet_v2
from spectrum import cmorlet

PATH_SAMPLE = '../tests/demo_data/demo_signal.csv'


def compare_wavelets(np_wavelets_v1, np_wavelets_v2, title_append=''):
    # Let's divide in real and imag banks
    real_v1, imag_v1 = np_wavelets_v1
    real_v2, imag_v2 = np_wavelets_v2

    # Visualize a certain scale
    which_scale = -1

    real_scale_v1 = real_v1[0, :, 0, which_scale]
    imag_scale_v1 = imag_v1[0, :, 0, which_scale]
    real_scale_v2 = real_v2[0, :, 0, which_scale]
    imag_scale_v2 = imag_v2[0, :, 0, which_scale]

    time_axis_v1 = np.arange(real_scale_v1.size) / fs
    time_axis_v1 = time_axis_v1 - np.mean(time_axis_v1)
    time_axis_v2 = np.arange(real_scale_v2.size) / fs
    time_axis_v2 = time_axis_v2 - np.mean(time_axis_v2)

    # Show wavelets
    fig, ax = plt.subplots(2, 1, sharex=True, dpi=100, figsize=(5, 5))
    ax[0].plot(time_axis_v1, real_scale_v1, label='Real part', linewidth=1)
    ax[0].plot(time_axis_v1, imag_scale_v1, label='Imag part', linewidth=1)
    ax[0].set_title('Target CMorlet Wavelet %s' % title_append)
    ax[0].legend()

    ax[1].plot(time_axis_v2, real_scale_v2, label='Real part', linewidth=1)
    ax[1].plot(time_axis_v2, imag_scale_v2, label='Imag part', linewidth=1)
    ax[1].set_title('Trainable CMorlet Wavelet %s' % title_append)
    ax[1].legend()

    plt.xlim([time_axis_v1[0], time_axis_v1[-1]])
    plt.show()


if __name__ == '__main__':
    fs = 200
    fb_true = [2.0]
    fb_initial = [0.5]
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
        fb_true,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        flattening=True,
        border_crop=0,
        stride=1)

    outputs_v2, wavelets_v2 = cmorlet_v2.compute_cwt(
        signal_ph,
        fb_initial,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        size_factor=3.0,
        flattening=True,
        border_crop=0,
        stride=1,
        trainable=True)

    # Add average pooling
    outputs_v1 = tf.layers.average_pooling2d(
        inputs=outputs_v1, pool_size=(8, 1), strides=(8, 1))
    outputs_v2 = tf.layers.average_pooling2d(
        inputs=outputs_v2, pool_size=(8, 1), strides=(8, 1))

    loss = tf.reduce_mean(tf.square(outputs_v1-outputs_v2))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    merged = tf.summary.merge_all()

    niters = 1000
    nstats = 50
    training = []
    iterations = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # See old wavelets
        np_wavelets_v2 = sess.run(
            wavelets_v2,
            feed_dict={signal_ph: demo_signal})
        compare_wavelets(
            np_wavelets_v1[0], np_wavelets_v2[0], title_append='(Initialization)')
        for i in range(niters):
            np_loss, _ = sess.run([loss, train_step],
                                  feed_dict={signal_ph: demo_signal})
            if i % nstats == 0:
                print('Iteration %d/%d, loss %1.6f' % (i+1, niters, np_loss))
            training.append(np_loss)
            iterations.append(i)

        fig = plt.figure(figsize=(5, 4), dpi=100)
        plt.plot(iterations, training, color='0.3')
        plt.title('Loss Evolution for Imitation Game')
        plt.xlabel('Iteration')
        plt.ylabel('Loss MSE')
        plt.show()

        # See new wavelets
        np_wavelets_v2 = sess.run(
            wavelets_v2,
            feed_dict={signal_ph: demo_signal})
        compare_wavelets(
            np_wavelets_v1[0], np_wavelets_v2[0], title_append='(End of training)')
