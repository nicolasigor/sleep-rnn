from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

detector_path = os.path.abspath('../..')
print(detector_path)
sys.path.append(detector_path)

from sleeprnn.data.loader import load_dataset
from sleeprnn.nn.spectrum import compute_sigma_band


if __name__ == '__main__':
    dataset = load_dataset('mass_ss')
    fs = dataset.fs
    x = dataset.get_subject_signal(subject_id=2, normalize_clip=False)
    y = dataset.get_subject_stamps(subject_id=2)
    which_stamp = 338  # 337  # 425
    context_size = 6
    center_sample = int(y[which_stamp, :].mean())
    start_sample = center_sample - (context_size // 2) * fs
    end_sample = center_sample + (context_size // 2) * fs
    demo_signal = x[start_sample:end_sample]

    print('Length of input:', demo_signal.size)
    fs = 200
    time_axis = np.arange(demo_signal.size) / fs

    # Build computational
    tf.reset_default_graph()
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, demo_signal.size],
                            name="feats_train_ph")
    outputs = compute_sigma_band(inputs, fs, ntaps=41)
    sess = tf.Session()

    results = sess.run(outputs, feed_dict={inputs: [demo_signal]})
    print(results.shape)

    # Show results
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=200)
    # Time-domain signals
    ax[0].plot(time_axis, demo_signal.flatten(), label='Original')
    ax[0].plot(time_axis, results.flatten(), label='Filtered')
    ax[0].set_title('Filter Test')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylim([-50, 50])
    ax[0].legend()

    ax[1].plot(time_axis, demo_signal.flatten() - results.flatten(), label='Residual')
    ax[1].set_title('Filter Test')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylim([-50, 50])
    ax[1].legend()

    plt.tight_layout()
    plt.show()
