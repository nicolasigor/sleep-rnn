"""Module that defines as base model class to manage neural networks."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import os

import tensorflow as tf

from utils import param_keys
from . import net_ops


class BaseModel(object):
    """ Base Model class to train and evaluate neural networks models.
    """
    def __init__(
            self,
            feat_shape,
            label_shape,
            params,
            logdir='logs'
    ):
        """ Constructor.

        Args:
            feat_shape: (iterable) Shape of the features of a single example.
            label_shape: (iterable) Shape of the labels of a single example.
            params: (dict) Dictionary of parameters to configure the model.
                See utils.model_keys for more details.
            logdir: (optional, string, defaults to 'logs') Directory of the
                model.
        """
        # Clean computational graph
        tf.reset_default_graph()

        # Save attributes
        self.feat_shape = list(feat_shape)
        self.label_shape = list(label_shape)
        self.logdir = logdir
        self.params = param_keys.default_params
        self.params.update(params)  # Overwrite defaults

        # Create directory of logs
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.ckptdir = os.path.join(self.logdir, 'model', 'ckpt')

        # --- Build model

        # Input placeholders
        self.feats_ph = tf.placeholder(
            tf.float32, shape=[None] + self.feat_shape, name='feats_ph')
        self.labels_ph = tf.placeholder(
            tf.int32, shape=[None] + self.label_shape, name='labels_ph')
        self.training_ph = tf.placeholder(tf.bool, name="training_ph")

        # Learning rate variable
        self.learning_rate = tf.Variable(
            self.params[param_keys.LEARNING_RATE], trainable=False, name='lr')

        # Input pipeline (single iterator)
        self.iterator = net_ops.get_iterator(
            self.feats_ph, self.labels_ph,
            batch_size=self.params[param_keys.BATCH_SIZE],
            shuffle_buffer_size=self.params[param_keys.SHUFFLE_BUFFER_SIZE],
            map_fn=self._map_fn,
            prefetch_buffer_size=self.params[param_keys.PREFETCH_BUFFER_SIZE])
        self.feats, self.labels = self.iterator.get_next()

        # Model prediction
        self.logits, self.probabilities = self._model_fn()

        # Evaluation metrics
        self.metrics_dict = self._metrics_fn()

        # Add training operations
        self.loss = self._loss_fn()
        self.train_step, self.reset_optimizer = self._optimizer_fn()

        # Fusion of all summaries
        self.merged = tf.summary.merge_all()

        # Tensorflow session for graph management
        self.sess = tf.Session()

        # Saver for checkpoints
        self.saver = tf.train.Saver()

        # Summary writers
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.logdir, 'train'))
        self.val_writer = tf.summary.FileWriter(
            os.path.join(self.logdir, 'val'))
        self.train_writer.add_graph(self.sess.graph)

        # Initialization op
        self.init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        # Save the parameters used to define this model
        with open(os.path.join(self.logdir, 'params.json'), 'w') as outfile:
            json.dump(self.params, outfile)

    def fit(self, x_train, y_train, x_val, y_val):
        # TODO: decay learning rate and early stopping
        """Fits the model to the training data"""
        iter_per_epoch = x_train.shape[0] // self.params[param_keys.BATCH_SIZE]
        niters = iter_per_epoch * self.params[param_keys.MAX_EPOCHS]
        nstats = iter_per_epoch
        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Max epochs %d, '
              'Training examples %d, Total iterations %d' %
              (self.params[param_keys.BATCH_SIZE],
               self.params[param_keys.MAX_EPOCHS], x_train.shape[0], niters))
        start_time = time.time()

        self.sess.run(self.init_op)
        self.sess.run(
            self.iterator.initializer,
            feed_dict={self.feats_ph: x_train, self.labels_ph: y_train})
        for it in range(1, niters+1):
            self.sess.run(self.train_step, feed_dict={self.training_ph: True})
            if it % nstats == 0 or it == 1:
                # Report stuff
                train_loss, summ = self.sess.run(
                    [self.loss, self.merged],
                    feed_dict={self.training_ph: False})
                self.train_writer.add_summary(summ, it)
                val_loss, summ = self.sess.run(
                    [self.loss, self.merged],
                    feed_dict={self.feats: x_val,
                               self.labels: y_val,
                               self.training_ph: False})
                self.val_writer.add_summary(summ, it)
                elapsed = time.time() - start_time
                print('It %6.0d/%d - loss train %1.6f val %1.6f - E.T. %1.4f s'
                      % (it, niters, train_loss, val_loss, elapsed))
        # Show final metrics
        train_metrics = self.sess.run(
            self.metrics_dict,
            feed_dict={self.feats: x_train, self.labels: y_train,
                       self.training_ph: False})
        val_metrics = self.sess.run(
            self.metrics_dict,
            feed_dict={self.feats: x_val, self.labels: y_val,
                       self.training_ph: False})
        print('Training metrics:\n')
        print(train_metrics)
        print('Validation metrics:\n')
        print(val_metrics)
        # Save fitted model checkpoint
        save_path = self.saver.save(self.sess, self.ckptdir)
        print('Model saved at %s' % save_path)
        return train_metrics, val_metrics

    def predict_proba(self, x):
        """Returns the probabilities predicted by the model for the input x."""
        y = self.sess.run(self.probabilities,
                          feed_dict={self.feats: x, self.training_ph: False})
        return y

    def load_checkpoint(self, ckptdir):
        """Loads variables from a checkpoint."""
        self.saver.restore(self.sess, ckptdir)

    def _map_fn(self, feat, label):
        """This method has to be implemented.

        This method is used to preprocess features and labels of single
        examples. Currently the method is a simple identity function.
        """
        return feat, label

    def _model_fn(self):
        """This method has to be implemented

        This method is used to evaluate the model with the inputs, and return
        logits and probabilities.
        """
        inputs = self.feats
        logits = None
        probabilities = None
        return logits, probabilities

    def _loss_fn(self):
        """This method has to be implemented

        This method is used to return the loss between the output of the
        model and the desired labels.
        """
        labels = self.labels
        logits = self.logits
        probabilities = self.probabilities
        loss = None
        return loss

    def _optimizer_fn(self):
        """This method has to be implemented

        This method is used to define the operation train_step that performs
        one training iteration to optimize the loss, and the reset optimizer
        operation that resets the variables of the optimizer, if any.
        """
        loss = self.loss
        train_step = None
        reset_optimizer_op = None
        return train_step, reset_optimizer_op

    def _metrics_fn(self):
        """This method has to be implemented

        This method is used to compute several useful metrics based on the
        model output and the desired labels. The metrics are stored in a metrics
        dictionary that is returned.
        """
        labels = self.labels
        probabilities = self.probabilities
        metrics_dict = {}
        return metrics_dict
