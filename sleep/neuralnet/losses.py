"""Module that defines losses to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sleep.utils import constants


def cross_entropy_loss_fn(logits, labels, class_weights):
    """Returns the cross-entropy loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i.
    """
    with tf.variable_scope(constants.CROSS_ENTROPY_LOSS):
        if class_weights is None:
            weights = 1.0
        elif class_weights == constants.BALANCED:
            n_negative = tf.reduce_sum(1 - labels)
            n_positive = tf.reduce_sum(labels)
            total = n_negative + n_positive
            weight_negative = n_positive / total
            weight_positive = n_negative / total
            class_weights = tf.stack([weight_negative, weight_positive], axis=0)
            weights = tf.gather(class_weights, labels)
        else:
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits,
                                                      weights=weights)
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def dice_loss_fn(probabilities, labels):
    """Returns 1-DICE loss to be minimized.

        Args:
            probabilities: (2d tensor) tensor of the probabilities of class 1
                with shape [batch, timelen]
            labels: (2d tensor) binary tensor of shape [batch, timelen]
        """
    with tf.variable_scope(constants.DICE_LOSS):
        labels = tf.to_float(labels)
        intersection = tf.reduce_sum(tf.multiply(probabilities, labels))
        size_prob = tf.reduce_sum(probabilities)
        size_labels = tf.reduce_sum(labels)
        dice = 2 * intersection / (size_prob + size_labels)
        loss = 1 - dice
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ
