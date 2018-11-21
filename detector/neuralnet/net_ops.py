"""net_ops.py: Module that defines useful operation to train a model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import constants


def get_iterator(
        feats_ph,
        labels_ph,
        batch_size,
        shuffle_buffer_size=0,
        map_fn=None,
        prefetch_buffer_size=0):
    """Builds efficient iterators for the training loop.

    Args:
        feats_ph: (tensor) Features placeholder
        labels_ph: (tensor) Labels placeholder
        batch_size: (int) Size of the minibatches
        shuffle_buffer_size: (Optional, int, defaults to 0) Size of the buffer
            to shuffle the data. If 0, no shuffle is applied.
        map_fn: (Optional, function, defaults to None) A function that
            preprocess the features and labels before passing them to the model.
        prefetch_buffer_size: (Optional, int, defaults to 0) Size of the buffer
            to prefetch the batches. If 0, no prefetch is applied.
    """
    with tf.name_scope("iterator"):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices((feats_ph, labels_ph))
            if shuffle_buffer_size > 0:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            dataset = dataset.repeat()
            if map_fn is not None:
                dataset = dataset.map(map_fn)
            dataset = dataset.batch(batch_size=batch_size)
        if prefetch_buffer_size > 0:
            dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
        iterator = dataset.make_initializable_iterator()
    return iterator


def get_global_iterator(handle_ph, iterator_1, iterator_2):
    """Builds a global iterator that can switch between two iterators.

    Args:
        handle_ph: (Tensor) Placeholder of type tf.string and shape [] that
            will be fed with the proper string_handle.
        iterator_1: (Iterator) First iterator that can be selected.
        iterator_2: (Iterator) Second iterator that can be selected.

    Returns:
        global_iterator: (Iterator) Iterator that will switch between iterator_1
            and iterator_2 according to the handle fed to handle_ph.
        handle_1: (String) Handle to select the iterator_1.
        handle_2: (String) Handle to select the iterator_2.
    """
    global_iterator = tf.data.Iterator.from_string_handle(
        handle_ph, iterator_1.output_types, iterator_2.output_shapes)
    with tf.Session() as sess:
        handle_1, handle_2 = sess.run(
            [iterator_1.string_handle(), iterator_2.string_handle()])
    return global_iterator, handle_1, handle_2


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
        tf.summary.scalar('loss', loss)
    return loss


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
        tf.summary.scalar('loss', loss)
    return loss


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
        tf.summary.scalar('grad_norm', grad_norm)

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
    return train_step, reset_optimizer_op


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
        tf.summary.scalar('grad_norm', grad_norm)

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
    return train_step, reset_optimizer_op
