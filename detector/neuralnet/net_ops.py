"""net_ops.py: Module that defines useful operation to train a model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import constants


def get_iterator(
        tensors_ph,
        batch_size,
        repeat=True,
        shuffle_buffer_size=0,
        map_fn=None,
        prefetch_buffer_size=0,
        name=None):
    """Builds efficient iterators for the training loop.

    Args:
        tensors_ph: (tensor) Input tensors placeholders
        batch_size: (int) Size of the minibatches
        repeat: (optional, boolean, defaults to True) whether to repeat
            ad infinitum the dataset or not.
        shuffle_buffer_size: (Optional, int, defaults to 0) Size of the buffer
            to shuffle the data. If 0, no shuffle is applied.
        map_fn: (Optional, function, defaults to None) A function that
            preprocess the features and labels before passing them to the model.
        prefetch_buffer_size: (Optional, int, defaults to 0) Size of the buffer
            to prefetch the batches. If 0, no prefetch is applied.
        name: (Optional, string, defaults to None) Name for the operation.
    """
    with tf.name_scope(name):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(tensors_ph)
            if shuffle_buffer_size > 0:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            if repeat:
                dataset = dataset.repeat()
            if map_fn is not None:
                dataset = dataset.map(map_fn)
            dataset = dataset.batch(batch_size=batch_size)
            if prefetch_buffer_size > 0:
                dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
            iterator = dataset.make_initializable_iterator()
    return iterator


def get_global_iterator(handle_ph, iterators_list, name=None):
    """Builds a global iterator that can switch between two iterators.

    Args:
        handle_ph: (Tensor) Placeholder of type tf.string and shape [] that
            will be fed with the proper string_handle.
        iterators_list: (list of Iterator) List of the iterators from where we
            can obtain inputs.
        name: (Optional, string, defaults to None) Name for the operation.

    Returns:
        global_iterator: (Iterator) Iterator that will switch between iterator_1
            and iterator_2 according to the handle fed to handle_ph.
    """
    with tf.name_scope(name):
        with tf.device('/cpu:0'):
            global_iterator = tf.data.Iterator.from_string_handle(
                handle_ph, iterators_list[0].output_types,
                iterators_list[0].output_shapes)
    return global_iterator


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
    with tf.name_scope(constants.CROSS_ENTROPY_LOSS):
        if class_weights is None:
            weights = 1.0
        elif class_weights == constants.BALANCED:
            n_negative = tf.reduce_sum(1 - labels)
            n_positive = tf.reduce_sum(labels)
            total = n_negative + n_positive
            weight_negative = n_positive / total
            weight_positive = n_negative / total
            class_weights = tf.constant([weight_negative, weight_positive])
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
    with tf.name_scope(constants.DICE_LOSS):
        intersection = tf.reduce_sum(tf.multiply(probabilities, labels))
        size_prob = tf.reduce_sum(tf.square(probabilities))
        size_labels = tf.reduce_sum(labels)
        dice = 2 * intersection / (size_prob + size_labels)
        loss = 1 - dice
        loss_summ = tf.summary.scalar('loss', loss)
    return loss, loss_summ


def adam_optimizer_fn(loss, learning_rate, clip_gradients, clip_norm):
    """Returns the optimizer operation to minimize the loss with Adam.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        clip_gradients: (boolean) Whether to clip gradient by the global norm.
        clip_norm: (float) Global norm to clip.
    """
    with tf.name_scope(constants.ADAM_OPTIMIZER):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(loss)

        gradients = [gv[0] for gv in gvs]
        grad_norm = tf.global_norm(gradients, name='gradient_norm')
        grad_norm_summ = tf.summary.scalar('grad_norm', grad_norm)

        if clip_gradients:
            gradients = tf.clip_by_global_norm(
                gradients,
                clip_norm,
                use_norm=grad_norm,
                name='clipping')
            variables = [gv[1] for gv in gvs]
            gvs = [(grad, var) for grad, var in zip(gradients, variables)]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For BN
        with tf.control_dependencies(update_ops):
            train_step = optimizer.apply_gradients(gvs)
        reset_optimizer_op = tf.variables_initializer(optimizer.variables())
    return train_step, reset_optimizer_op, grad_norm_summ


def sgd_optimizer_fn(loss, learning_rate, momentum, clip_gradients, clip_norm):
    """Returns the optimizer operation to minimize the loss with SGD with
    Nesterov momentum.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        momentum: (Optional, float) momentum for the optimizer.
        clip_gradients: (boolean) Whether to clip gradient by the global norm.
        clip_norm: (float) Global norm to clip.
    """
    with tf.name_scope(constants.SGD_OPTIMIZER):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=True)
        gvs = optimizer.compute_gradients(loss)

        gradients = [gv[0] for gv in gvs]
        grad_norm = tf.global_norm(gradients, name='gradient_norm')
        grad_norm_summ = tf.summary.scalar('grad_norm', grad_norm)

        if clip_gradients:
            gradients = tf.clip_by_global_norm(
                gradients,
                clip_norm,
                use_norm=grad_norm,
                name='clipping')
            variables = [gv[1] for gv in gvs]
            gvs = [(grad, var) for grad, var in zip(gradients, variables)]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For BN
        with tf.control_dependencies(update_ops):
            train_step = optimizer.apply_gradients(gvs)
        reset_optimizer_op = tf.variables_initializer(optimizer.variables())
    return train_step, reset_optimizer_op, grad_norm_summ


def confusion_matrix(logits, labels):
    """Returns TP, FP and FN"""
    with tf.name_scope('confusion_matrix'):
        predictions_sparse = tf.argmax(logits, axis=-1)
        labels_zero = tf.equal(labels, tf.zeros_like(labels))
        labels_one = tf.equal(labels, tf.ones_like(labels))
        predictions_zero = tf.equal(
            predictions_sparse, tf.zeros_like(predictions_sparse))
        predictions_one = tf.equal(
            predictions_sparse, tf.ones_like(predictions_sparse))

        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(labels_one, predictions_one), "float"))
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(labels_zero, predictions_one), "float"))
        fn = tf.reduce_sum(
            tf.cast(tf.logical_and(labels_one, predictions_zero), "float"))
    return tp, fp, fn


def precision_recall_f1score(tp, fp, fn):
    """Return Precision, Recall, and F1-Score metrics."""
    with tf.name_scope('precision'):
        # Edge case: no detections -> precision 1
        precision = tf.cond(
            pred=tf.equal((tp + fp), 0),
            true_fn=lambda: 1.0,
            false_fn=lambda: tp / (tp + fp)
        )
    with tf.name_scope('recall'):
        # Edge case: no marks -> recall 1
        recall = tf.cond(
            pred=tf.equal((tp + fn), 0),
            true_fn=lambda: 1.0,
            false_fn=lambda: tp / (tp + fn)
        )
    with tf.name_scope('f1_score'):
        # Edge case: precision and recall 0 -> f1 score 0
        f1_score = tf.cond(
            pred=tf.equal((precision + recall), 0),
            true_fn=lambda: 0.0,
            false_fn=lambda: 2 * precision * recall / (precision + recall)
        )
    return precision, recall, f1_score
