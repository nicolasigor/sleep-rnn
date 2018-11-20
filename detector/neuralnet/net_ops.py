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


def loss_fn(logits, labels, class_weights):
    """Returns the loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timesteps]
        class_weights: (1d array_like) Array of shape [2,] that contains
            the class weights. If None, no weights are applied.
    """
    with tf.name_scope('loss'):
        if class_weights is None:
            weights = 1.0
        else:
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits,
                                                      weights=weights)
        tf.summary.scalar('loss', loss)
    return loss


def optimizer_fn(loss, learning_rate, clip_gradients=False):
    """Returns the optimizer operation to minimize the loss with Adam.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (int) learning rate for the optimizer
        clip_gradients: (Optional, boolean, defaults to False) Whether to
            clip the gradient by the global norm.
    """
    with tf.name_scope("optimizer"):
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
