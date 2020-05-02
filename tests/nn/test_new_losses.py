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

from sleeprnn.nn import losses


def plot_with_opt(ax, x, y, label):
    opt_idx = np.argmin(y)
    ax.plot(x, y, label=label, marker='o', markevery=[opt_idx])
    return ax


def prepare_plot(ax, xlim=[-0.01, 1.01], ylim=[-0.1, 0.1]):
    ax.set_xlabel('Probability of Class 1')
    ax.legend(fontsize=9, bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


if __name__ == '__main__':
    true_class = 0
    beta_eps_gamma = [0.5, 0.1, 1.5]
    min_val = 1e-4
    n_points = 300

    # Preliminaries
    pred_1 = np.linspace(
        min_val, 1 - min_val,
        num=n_points, endpoint=True, dtype=np.float32)
    logits_0 = pred_1 * 0
    logits_1 = np.log(pred_1 / (1 - pred_1)).astype(np.float32)
    logits = np.stack([logits_0, logits_1], axis=1).reshape((-1, 1, 1, 2))
    single_label = np.array([true_class], dtype=np.int32).reshape(1, 1)
    print('Single label', single_label.shape, single_label.dtype)
    print('Logits', logits.shape, logits.dtype)

    # ---------------------------------------------------------------------
    # Test losses
    beta = beta_eps_gamma[0]
    gamma = beta_eps_gamma[2]
    eps = beta_eps_gamma[1]
    tf.reset_default_graph()
    logits_ph = tf.placeholder(shape=[1, 1, 2], dtype=tf.float32)
    # Losses
    tf_xent, _ = losses.cross_entropy_loss_fn(
        logits_ph, single_label, None)
    tf_xent_negent, _ = losses.cross_entropy_negentropy_loss_fn(
        logits_ph, single_label, None, beta)
    tf_xent_smooth, _ = losses.cross_entropy_smoothing_loss_fn(
        logits_ph, single_label, None, eps)
    tf_focal, _ = losses.focal_loss_fn(
        logits_ph, single_label, None, gamma)
    tf_xent_hard_clip, _ = losses.cross_entropy_hard_clip_loss_fn(
        logits_ph, single_label, None, eps)
    tf_xent_smooth_clip, _ = losses.cross_entropy_smoothing_clip_loss_fn(
        logits_ph, single_label, None, eps)
    sess = tf.Session()
    tf.global_variables_initializer()

    def evaluate(tensor_name):
        loss = np.array([
            sess.run(tensor_name, feed_dict={logits_ph: single_logit})
            for single_logit in logits
        ])
        return loss

    # Evaluation
    np_xent = evaluate(tf_xent)
    np_xent_negent = evaluate(tf_xent_negent)
    np_xent_smooth = evaluate(tf_xent_smooth)
    np_focal = evaluate(tf_focal)
    np_xent_hard_clip = evaluate(tf_xent_hard_clip)
    np_xent_smooth_clip = evaluate(tf_xent_smooth_clip)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3), dpi=100)
    ax = plot_with_opt(ax, pred_1, np_xent, 'xEnt-%d' % true_class)
    ax = plot_with_opt(ax, pred_1, np_xent_negent - np_xent_negent.min(), 'xEnt-%d + NegEnt (beta %1.1f)' % (true_class, beta))
    ax = plot_with_opt(ax, pred_1, np_xent_smooth, 'xEnt-%d + Smooth (eps %1.2f)' % (true_class, eps))
    ax = plot_with_opt(ax, pred_1, np_focal, 'Focal-%d (gamma %1.1f)' % (true_class, gamma))
    ax = plot_with_opt(ax, pred_1, np_xent_hard_clip, 'xEnt-%d + HardClip (eps %1.2f)' % (true_class, eps))
    ax = plot_with_opt(ax, pred_1, np_xent_smooth_clip, 'xEnt-%d + SmoothClip (eps %1.2f)' % (true_class, eps))
    ax = prepare_plot(ax, xlim=[-0.01, 0.4], ylim=[-0.01, 0.1])
    plt.show()


