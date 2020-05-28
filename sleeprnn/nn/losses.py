"""Module that defines losses to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from sleeprnn.common import constants


def get_weights(logits, labels, class_weights):
    with tf.variable_scope('loss_weights'):
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
            print('Balancing with weights', class_weights)
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)
    return weights


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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
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
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def mod_focal_loss_fn(logits, labels, class_weights, gamma, mis_weight):
    """Returns the MODIFIED focal loss to be minimized.

    "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    If p is probability of correct class, the original Focal loss weights the
    cross-entropy loss with:
    w = (1-p) ^ gamma

    The modified version weights the cross-entropy loss with:
    w = 1 + (mis_weight - 1) * (1-p) ^ gamma

    If mis_weight = 1 or gamma = 0, this reduces to w = 1 for all samples
    (regular cross-entropy)

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
        mis_weight: Weight for misclassification (greater or equal to 1).
    """
    print(
        'Using MODIFIED Focal Loss (gamma = %1.4f, mis_weight = %1.4f)' % (
            gamma, mis_weight))
    with tf.variable_scope(constants.MOD_FOCAL_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        # Apply focusing parameter
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2)  # output shape [batch, time]

        focal_term = (1.0 - proba_correct_class) ** gamma
        error_weight = 1.0 + (mis_weight - 1.0) * focal_term
        loss = error_weight * loss
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


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
    print('Using Focal Loss (gamma = %1.4f)' % gamma)
    with tf.variable_scope(constants.FOCAL_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        # Apply focusing parameter
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2)  # output shape [batch, time]
        loss = ((1.0 - proba_correct_class) ** gamma) * loss
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def cross_entropy_negentropy_loss_fn(logits, labels, class_weights, beta):
    print('Using Cross Entropy Loss with Entropy Regularization (beta = %1.4f)' % beta)
    with tf.variable_scope(constants.CROSS_ENTROPY_NEG_ENTROPY_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        # Regularization
        probabilities = tf.nn.softmax(logits)
        neg_entropy = tf.reduce_sum(
            probabilities * tf.log(probabilities + 1e-6), axis=2)  # output shape [batch, time]
        loss = loss + beta * neg_entropy
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def cross_entropy_smoothing_loss_fn(logits, labels, class_weights, epsilon):
    print('Using Cross Entropy Loss with Label Smoothing (eps = %1.4f)' % epsilon)
    with tf.variable_scope(constants.CROSS_ENTROPY_SMOOTHING_LOSS):
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        smooth_labels = labels_onehot * (1.0 - 2.0 * epsilon) + epsilon
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=smooth_labels,
            logits=logits)
        offset = -tf.reduce_sum(smooth_labels * tf.log(smooth_labels + 1e-6), axis=2)
        loss = loss - offset
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def cross_entropy_hard_clip_loss_fn(logits, labels, class_weights, epsilon):
    print('Using Cross Entropy Loss with Hard Clip (eps = %1.4f)' % epsilon)
    with tf.variable_scope(constants.CROSS_ENTROPY_HARD_CLIP_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        # Clip
        offset = -tf.log(1.0 - epsilon)
        loss = tf.nn.relu(loss - offset)
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def cross_entropy_smoothing_clip_loss_fn(logits, labels, class_weights, epsilon):
    print('Using Cross Entropy Loss with Label Smoothing and Clip (eps = %1.4f)' % epsilon)
    with tf.variable_scope(constants.CROSS_ENTROPY_SMOOTHING_CLIP_LOSS):
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        smooth_labels = labels_onehot * (1.0 - 2.0 * epsilon) + epsilon
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=smooth_labels,
            logits=logits)
        offset = -tf.reduce_sum(smooth_labels * tf.log(smooth_labels + 1e-6), axis=2)
        loss = loss - offset
        # Clip
        probabilities = tf.nn.softmax(logits)
        distance_to_hard_label = tf.reduce_sum((probabilities - labels_onehot) ** 2, axis=2)
        distance_thr = 2 * (epsilon ** 2)
        mask = tf.cast(tf.math.greater(distance_to_hard_label, distance_thr), tf.float32)
        loss = loss * mask
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def worst_mining_loss_fn(logits, labels, factor_negative, min_negative):
    """Returns the balanced cross-entropy loss to be minimized by worst
    negative mining

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        factor_negative: (int) Ratio of negatives / positives
        min_negative: (int) minimum number of negatives examples that are
            kept when balancing.

    """
    print('Using Worst Mining Loss (individual version)')
    with tf.variable_scope(constants.WORST_MINING_LOSS):
        print('Min negative:', min_negative)
        print('Factor negative:', factor_negative)
        min_negative = tf.cast(min_negative, tf.float32)
        factor_negative = tf.cast(factor_negative, tf.float32)
        batch_size = 32
        print("-----------------------------------")
        print("-----------------------------------")
        print("AT LOSS, BATCH SIZE HARDCODED AS %d" % batch_size)
        print("-----------------------------------")
        print("-----------------------------------")

        # Unstack batch
        logits_list = tf.unstack(logits, axis=0, num=batch_size)
        labels_list = tf.unstack(labels, axis=0, num=batch_size)
        sum_loss_batch = 0
        for s_logits, s_labels in zip(logits_list, labels_list):
            # # First we flatten the vectors
            # logits = tf.reshape(logits, [-1, 2])  # [n_outputs, 2]
            # labels = tf.reshape(labels, [-1])  # [n_outputs]

            # Number of positives and negatives
            labels_float = tf.cast(s_labels, tf.float32)
            n_negative = tf.reduce_sum(1 - labels_float)
            n_positive = tf.reduce_sum(labels_float)

            # Compute number of negatives to keep
            feasible_min_negative = tf.minimum(min_negative, n_negative)
            n_negative_to_keep = tf.minimum(
                factor_negative * n_positive,
                n_negative)
            n_negative_to_keep = tf.maximum(
                n_negative_to_keep,
                feasible_min_negative)

            # Compute loss
            loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=s_labels,
                logits=s_logits)
            loss_positive = loss_raw * labels_float
            loss_negative = loss_raw * (1 - labels_float)
            loss_negative_to_keep, _ = tf.nn.top_k(
                loss_negative,
                k=tf.cast(n_negative_to_keep, tf.int32))
            sum_loss = tf.reduce_sum(loss_positive) + tf.reduce_sum(loss_negative_to_keep)
            n_examples = n_positive + n_negative_to_keep
            s_loss = sum_loss / n_examples

            # Add to batch sum
            sum_loss_batch = sum_loss_batch + s_loss

        # Loss is mean of single losses
        loss = sum_loss_batch / batch_size
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def worst_mining_v2_loss_fn(logits, labels, factor_negative, min_negative):
    """Returns the balanced cross-entropy loss to be minimized by worst
    negative mining. L+ and L- are averaged independently.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        factor_negative: (int) Ratio of negatives / positives
        min_negative: (int) minimum number of negatives examples that are
            kept when balancing.

    """
    print('Using Worst Mining Loss V2 (pos and neg parts averaged before sum)')
    with tf.variable_scope(constants.WORST_MINING_V2_LOSS):
        print('Min negative:', min_negative)
        print('Factor negative:', factor_negative)

        # First we flatten the vectors
        logits = tf.reshape(logits, [-1, 2])  # [n_outputs, 2]
        labels = tf.reshape(labels, [-1])  # [n_outputs]

        labels_float = tf.cast(labels, tf.float32)

        n_negative = tf.reduce_sum(1 - labels_float)
        n_positive = tf.reduce_sum(labels_float)

        n_negative_to_keep = tf.minimum(
            factor_negative * n_positive,
            n_negative
        )
        n_negative_to_keep = tf.maximum(
            n_negative_to_keep,
            min_negative
        )

        loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)

        loss_positive = loss_raw * labels_float

        loss_negative = loss_raw * (1 - labels_float)
        loss_negative_to_keep, _ = tf.nn.top_k(
            loss_negative,
            k=tf.cast(n_negative_to_keep, tf.int32)
        )

        loss_positive_norm = tf.reduce_sum(loss_positive) / (n_positive + 1e-8)
        loss_negative_norm = tf.reduce_sum(loss_negative_to_keep) / n_negative_to_keep

        loss = loss_positive_norm + loss_negative_norm

        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ
