"""models.py: Module that defines trainable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from . import networks
from . import net_ops
from utils.constants import CHANNELS_LAST, CHANNELS_FIRST
from utils.constants import PAD_SAME, PAD_VALID
from utils.constants import BN, BN_RENORM
from utils.constants import MAXPOOL, AVGPOOL
from utils.constants import SEQUENCE_DROP, REGULAR_DROP
from utils.constants import UNIDIRECTIONAL, BIDIRECTIONAL
from utils.constants import ERROR_INVALID


# TODO: make a Base Class BaseModel


class BaseModel(object):
    def __init__(
            self,
            feat_shape,
            label_shape,
            init_learning_rate,
            class_weights,
            batch_size,
            clip_gradients,
            max_epochs,
            logdir,
            params):
        # Clean computational graph
        tf.reset_default_graph()

        # Save attributes
        self.feat_shape = list(feat_shape)
        self.label_shape = list(label_shape)
        self.init_learning_rate = init_learning_rate
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.clip_gradients=clip_gradients
        self.max_epochs = max_epochs
        self.logdir = logdir
        self.params = params  # Holds variables for the children

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

        # Input pipeline (single iterator)
        self.iterator = net_ops.input_pipeline(
            self.feats_ph, self.labels_ph, self.batch_size, self._map_fn)
        self.feats, self.labels = self.iterator.get_next()

        # Model prediction
        self.logits, self.probabilities = self._model_fn(self.feats)

        # Evaluation metrics
        self.metrics_dict = self._metrics_fn(self.probabilities, self.labels)

        # Add training operations
        self.loss = net_ops.loss_fn(
            self.logits, self.labels, self.class_weights)
        self.learning_rate = tf.Variable(
            self.init_learning_rate, trainable=False)
        self.train_step, self.reset_optimizer = net_ops.optimizer_fn(
            self.loss, self.learning_rate, self.clip_gradients)

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

    # TODO: Reportar metricas bs
    def fit(self, x_train, y_train, x_val, y_val):
        """Fits the model to the training data"""
        iter_per_epoch = x_train.shape[0] // self.batch_size
        niters = iter_per_epoch * self.max_epochs
        nstats = iter_per_epoch
        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Max epochs %d, '
              'Training examples %d, Total iterations %d' %
              (self.batch_size, self.max_epochs, x_train.shape[0], niters))
        start_time = time.time()

        self.sess.run(self.init_op)
        self.sess.run(self.iterator.initializer,
                      feed_dict={self.feats_ph: x_train,
                                 self.labels_ph: y_train})
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
                print('It %6.0d/%d - loss train %1.6f test %1.6f - E.T. %1.4f s'
                      % (it, niters, train_loss, val_loss, elapsed))
        # Save fitted model checkpoint
        save_path = self.saver.save(self.sess, self.ckptdir, global_step=niters)
        print('Model saved at %s' % save_path)

    def predict_proba(self, x):
        y = self.sess.run(self.probabilities,
                          feed_dict={self.feats: x, self.training_ph: False})
        return y

    def _model_fn(self, inputs):
        """This method has to be implemented"""
        logits = None
        probabilities = None
        return logits, probabilities

    def _map_fn(self, feat, label):
        """This method has to be implemented if desired."""
        return feat, label

    def _metrics_fn(self, probabilities, labels):
        """This method has to be implemented"""
        metrics_dict = {}
        return metrics_dict


#class MorletConvBLSTM(BaseModel):
#    pass
    # Use self.params for these methods.

   # def _model_fn(self, inputs):
   #     logits, probabilities = networks.cmorlet_conv_blstm_net(
   #
   #     )
   #     return logits, probabilities

   # def _map_fn(self, feat, label):
   #     """Random cropping"""
   #     crop_size = self.p["crop_size"]
   #     time_stride = self.p["time_stride"]
   #     border_size = self.p["border_size"]
   #     # Random crop
   #     label_cast = tf.cast(label, dtype=tf.float32)
   #     stack = tf.stack([feat, label_cast], axis=0)
   #     stack_crop = tf.random_crop(stack, [2, crop_size])
   #     feat = stack_crop[0, :]
   #     # Throw first and last second for labels, skipping steps
   #     label_cast = stack_crop[1, border_size:-border_size:time_stride]
   #     label = tf.cast(label_cast, dtype=tf.int32)
   #     return feat, label


# TODO: implement spline model
#class SplineConvBLSTM(BaseModel):
#    pass




# def update_learning_rate(self, global_step):
#        self.sess.run(tf.assign(self.learning_rate,
#0.04 / (2.0 ** (global_step // self.params["iterations_to_update_learning_rate"]))))