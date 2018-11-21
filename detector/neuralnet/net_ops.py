"""net_ops.py: Module that defines useful operation to train a model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def input_pipeline(
        feats_ph,
        labels_ph,
        batch_size,
        map_fn=None):
    """Builds an input pipeline with efficient iterators for the training loop.

    Args:
        feats_ph: (tensor) Features placeholder
        labels_ph: (tensor) Labels placeholder
        batch_size: (int) Size of the minibatches
        map_fn: (Optional, function, defaults to None) A function that
            preprocess the features and labels before passing them to the model.
    """
    with tf.name_scope("iterator"):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices((feats_ph, labels_ph))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.repeat()
            if map_fn is not None:
                dataset = dataset.map(map_fn)
            dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=2)
        iterator = dataset.make_initializable_iterator()
    return iterator


def cross_entropy_loss_fn(logits, labels, class_weights):
    """Returns the cross-entropy loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, 'balanced', array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If 'balanced', the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i.
    """
    with tf.name_scope('crossentropy_loss'):
        if class_weights is None:
            weights = 1.0
        elif class_weights == 'balanced':
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
    with tf.name_scope('dice_loss'):
        intersection = tf.reduce_sum(tf.multiply(probabilities, labels))
        size_prob = tf.reduce_sum(tf.square(probabilities))
        size_labels = tf.reduce_sum(labels)
        dice = 2 * intersection / (size_prob + size_labels)
        loss = 1 - dice
        tf.summary.scalar('loss', loss)
    return loss


def adam_optimizer_fn(loss, learning_rate, clip_gradients=False):
    """Returns the optimizer operation to minimize the loss with Adam.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        clip_gradients: (Optional, boolean, defaults to False) Whether to
            clip the gradient by the global norm.
    """
    with tf.name_scope("adam_optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(loss)

        gradients = [gv[0] for gv in gvs]
        grad_norm = tf.global_norm(gradients, name='gradient_norm')
        tf.summary.scalar('grad_norm', grad_norm)

        if clip_gradients:
            clip_norm = 5
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


def sgd_optimizer_fn(loss, learning_rate, momentum=0.9, clip_gradients=False):
    """Returns the optimizer operation to minimize the loss with SGD with
    Nesterov momentum.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        momentum: (Optional, float, defaults to 0.9) momentum for the optimizer.
        clip_gradients: (Optional, boolean, defaults to False) Whether to
            clip the gradient by the global norm.
    """
    with tf.name_scope("sgd_optimizer"):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=True)
        gvs = optimizer.compute_gradients(loss)

        gradients = [gv[0] for gv in gvs]
        grad_norm = tf.global_norm(gradients, name='gradient_norm')
        tf.summary.scalar('grad_norm', grad_norm)

        if clip_gradients:
            clip_norm = 5
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
    return train_step
