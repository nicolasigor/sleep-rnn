"""Module that defines losses to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from sleeprnn.common import constants


def cross_entropy_loss_fn(logits, labels, class_weights):
    """Returns the cross-entropy loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
    """
    print('Using Cross Entropy Loss')
    with tf.variable_scope(constants.CROSS_ENTROPY_LOSS):
        if class_weights is None:
            print('No weight balancing')
            class_weights = tf.constant([1.0, 1.0], dtype=tf.float32)
            weights = tf.gather(class_weights, labels)
        elif class_weights == constants.BALANCED:
            print('Class freq as weight to balance')
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)
            total = n_negative + n_positive
            weight_negative = n_positive / total
            weight_positive = n_negative / total
            class_weights = tf.stack([weight_negative, weight_positive], axis=0)
            weights = tf.gather(class_weights, labels)
        elif class_weights == constants.BALANCED_DROP:
            print('Random negative class dropping to balance')
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)
            p_drop_negative = (n_negative - n_positive) / n_negative
            random_mask = tf.random.uniform(
                tf.shape(labels),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32)
            random_mask = random_mask + tf.cast(labels, tf.float32)
            weights = tf.cast(
                tf.math.greater(random_mask, p_drop_negative), tf.float32)
        elif class_weights == constants.BALANCED_DROP_V2:
            print('Random negative class dropping to balance (V2)')
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)

            # Add negative labels on the neighborhood of positive ones
            neighbor_radius = 4

            labels_2d = tf.expand_dims(tf.cast(labels, tf.float32), 1)
            labels_2d = tf.expand_dims(labels_2d, 3)
            spread_labels = tf.nn.conv2d(
                labels_2d, filter=np.ones((1, 2 * neighbor_radius + 1, 1, 1)),
                strides=[1, 1, 1, 1], padding='SAME')
            spread_labels = tf.squeeze(spread_labels, [1, 3])
            spread_labels = tf.cast(
                tf.math.greater(spread_labels, 0), tf.float32)
            n_negative_already_added = tf.reduce_sum(spread_labels) - n_positive
            n_negative_to_add = tf.nn.relu(
                n_positive - n_negative_already_added)
            p_keep_negative = n_negative_to_add / (
                        n_negative + 1e-3)  # in (0, 1)
            p_drop_negative = 1 - p_keep_negative

            random_mask = tf.random.uniform(
                tf.shape(labels),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32)
            random_mask = random_mask + spread_labels
            weights = tf.cast(
                tf.math.greater(random_mask, p_drop_negative), tf.float32)
        else:
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)

        # loss = tf.losses.sparse_softmax_cross_entropy(
        #     labels=labels, logits=logits, weights=weights)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        # Weighted loss
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)

        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def dice_loss_fn(probabilities, labels):
    """Returns 1-DICE loss to be minimized.

        Args:
            probabilities: (2d tensor) tensor of the probabilities of class 1
                with shape [batch, timelen]
            labels: (2d tensor) binary tensor of shape [batch, timelen]
        """
    print('Using DICE Loss')
    with tf.variable_scope(constants.DICE_LOSS):
        labels = tf.to_float(labels)
        intersection = tf.reduce_sum(tf.multiply(probabilities, labels))
        size_prob = tf.reduce_sum(probabilities)
        size_labels = tf.reduce_sum(labels)
        dice = 2 * intersection / (size_prob + size_labels)
        loss = 1 - dice
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


# TODO: Implement Focal Loss
def focal_loss_fn(logits, labels, class_weights, gamma):
    """Returns the focal loss to be minimized.

    "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
        gamma: Focusing parameter (Non-negative)
    """
    print('Using Focal Loss')
    with tf.variable_scope(constants.FOCAL_LOSS):
        if class_weights is None:
            print('No weight balancing')
            class_weights = tf.constant([1.0, 1.0], dtype=tf.float32)
            weights = tf.gather(class_weights, labels)
        elif class_weights == constants.BALANCED:
            print('Class freq as weight to balance')
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)
            total = n_negative + n_positive
            weight_negative = n_positive / total
            weight_positive = n_negative / total
            class_weights = tf.stack([weight_negative, weight_positive], axis=0)
            weights = tf.gather(class_weights, labels)
        elif class_weights == constants.BALANCED_DROP:
            print('Random negative class dropping to balance')
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)
            p_drop_negative = (n_negative - n_positive) / n_negative
            random_mask = tf.random.uniform(
                tf.shape(labels),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32)
            random_mask = random_mask + tf.cast(labels, tf.float32)
            weights = tf.cast(
                tf.math.greater(random_mask, p_drop_negative), tf.float32)
        elif class_weights == constants.BALANCED_DROP_V2:
            print('Random negative class dropping to balance (V2)')
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)

            # Add negative labels on the neighborhood of positive ones
            neighbor_radius = 4

            labels_2d = tf.expand_dims(tf.cast(labels, tf.float32), 1)
            labels_2d = tf.expand_dims(labels_2d, 3)
            spread_labels = tf.nn.conv2d(
                labels_2d, filter=np.ones((1, 2 * neighbor_radius + 1, 1, 1)),
                strides=[1, 1, 1, 1], padding='SAME')
            spread_labels = tf.squeeze(spread_labels, [1, 3])
            spread_labels = tf.cast(
                tf.math.greater(spread_labels, 0), tf.float32)
            n_negative_already_added = tf.reduce_sum(spread_labels) - n_positive
            n_negative_to_add = tf.nn.relu(
                n_positive - n_negative_already_added)
            p_keep_negative = n_negative_to_add / (
                        n_negative + 1e-3)  # in (0, 1)
            p_drop_negative = 1 - p_keep_negative

            random_mask = tf.random.uniform(
                tf.shape(labels),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32)
            random_mask = random_mask + spread_labels
            weights = tf.cast(
                tf.math.greater(random_mask, p_drop_negative), tf.float32)
        else:
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)

        # loss = tf.losses.sparse_softmax_cross_entropy(
        #     labels=labels, logits=logits, weights=weights)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)


        # Weighted loss
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)

        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ
