"""Module that defines as base model class to manage neural networks."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

from sleeprnn.common import pkeys
from sleeprnn.common import constants
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection.predicted_dataset import PredictedDataset
from sleeprnn.data.utils import pages2seq
from . import feeding

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_TO_PROJECT = os.path.abspath(os.path.join(PATH_THIS_DIR, '..'))

# Summaries keys
KEY_LOSS = 'loss'
KEY_GRAD_NORM = 'grad_norm'
KEY_BATCH_METRICS = 'batch_metrics'
KEY_EVAL_METRICS = 'eval_metrics'


class BaseModelMod(object):
    """ Base Model class to train and evaluate neural networks models.
    """
    def __init__(
            self,
            feat_train_shape,
            label_train_shape,
            feat_eval_shape,
            label_eval_shape,
            params,
            logdir='logs'
    ):
        """ Constructor.

        Args:
            feat_train_shape: (iterable) Shape of the features of a single
                example that is the input to the training iterator.
            label_train_shape: (iterable) Shape of the labels of a single
                example that is the input to the training iterator.
            feat_eval_shape: (iterable) Shape of the features of a single
                example that is the input to the evaluation iterator.
            label_eval_shape: (iterable) Shape of the labels of a single
                example that is the input to the evaluation iterator.
            params: (dict) Dictionary of parameters to configure the model.
                See common.model_keys for more details.
            logdir: (optional, string, defaults to 'logs') Directory of the
                model. This path can be absolute, or relative to project root.
        """
        # Clean computational graph
        tf.reset_default_graph()

        # Save attributes
        self.feat_train_shape = list(feat_train_shape)
        self.label_train_shape = list(label_train_shape)
        self.feat_eval_shape = list(feat_eval_shape)
        self.label_eval_shape = list(label_eval_shape)
        if os.path.isabs(logdir):
            self.logdir = logdir
        else:
            self.logdir = os.path.join(PATH_TO_PROJECT, logdir)
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults

        # Create directory of logs
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.ckptdir = os.path.join(self.logdir, 'model', 'ckpt')

        # --- Build model

        # Input placeholders
        with tf.variable_scope('inputs_ph'):
            self.handle_ph = tf.placeholder(
                tf.string, shape=[], name='handle_ph')
            self.feats_train_1_ph = tf.placeholder(
                tf.float32, shape=[None] + self.feat_train_shape,
                name='feats_train_1_ph')
            self.labels_train_1_ph = tf.placeholder(
                tf.int32, shape=[None] + self.label_train_shape,
                name='labels_train_1_ph')
            self.sub_ids_train_1_ph = tf.placeholder(
                tf.int32, shape=[None], name='sub_ids_train_1_ph')

            self.feats_train_2_ph = tf.placeholder(
                tf.float32, shape=[None] + self.feat_train_shape,
                name='feats_train_2_ph')
            self.labels_train_2_ph = tf.placeholder(
                tf.int32, shape=[None] + self.label_train_shape,
                name='labels_train_2_ph')
            self.sub_ids_train_2_ph = tf.placeholder(
                tf.int32, shape=[None], name='sub_ids_train_2_ph')

            self.feats_eval_ph = tf.placeholder(
                tf.float32, shape=[None] + self.feat_eval_shape,
                name='feats_eval_ph')
            self.labels_eval_ph = tf.placeholder(
                tf.int32, shape=[None] + self.label_eval_shape,
                name='labels_eval_ph')
            self.sub_ids_eval_ph = tf.placeholder(
                tf.int32, shape=[None], name='sub_ids_eval_ph')

            self.training_ph = tf.placeholder(tf.bool, name="training_ph")

        # Learning rate variable
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(
                self.params[pkeys.LEARNING_RATE], trainable=False,
                name='lr')
            self.lr_summ = tf.summary.scalar('lr', self.learning_rate)
            self.lr_updates = 0

        with tf.variable_scope('feeding'):
            # Training iterator
            self.iterator_train = feeding.get_iterator_splitted(
                (self.feats_train_1_ph, self.labels_train_1_ph, self.sub_ids_train_1_ph),
                (self.feats_train_2_ph, self.labels_train_2_ph, self.sub_ids_train_2_ph),
                batch_size=self.params[pkeys.BATCH_SIZE],
                shuffle_buffer_size=self.params[pkeys.SHUFFLE_BUFFER_SIZE],
                map_fn=self._train_map_fn,
                prefetch_buffer_size=self.params[pkeys.PREFETCH_BUFFER_SIZE],
                name='iter_train')

            # Evaluation iterator
            self.iterator_eval = feeding.get_iterator(
                (self.feats_eval_ph, self.labels_eval_ph, self.sub_ids_eval_ph),
                batch_size=self.params[pkeys.BATCH_SIZE],
                shuffle_buffer_size=0,
                map_fn=self._eval_map_fn,
                prefetch_buffer_size=1,
                name='iter_eval')

            # Global iterator
            iterators_list = [self.iterator_train, self.iterator_eval]
            self.iterator = feeding.get_global_iterator(
                    self.handle_ph, iterators_list,
                    name='iters')
            self.feats, self.labels, self.sub_ids = self.iterator.get_next()

        # Model prediction
        self.logits, self.probabilities, self.cwt_prebn = self._model_fn()

        # Add training operations
        self.loss, self.loss_sum = self._loss_fn()
        self.train_step, self.reset_optimizer, self.grad_norm_summ = self._optimizer_fn()

        # BN after CWT stuff
        model_version = params[pkeys.MODEL_VERSION]
        if model_version == constants.V1:
            model_name = 'model_v1'
        elif model_version == constants.V4:
            model_name = 'model_v4'
        else:
            model_name = 'dummy'

        # self.personalize = tf.get_collection(
        #     tf.GraphKeys.UPDATE_OPS,
        #     scope='%s/spectrum' % model_name)

        self.ind_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/spectrum' % model_name)
        self.ind_variables = [var for var in self.ind_variables if 'moving' in var.name]
        # print(self.ind_variables)

        # Evaluation metrics
        self.batch_metrics_dict, self.batch_metrics_summ = self._batch_metrics_fn()
        self.eval_metrics_dict, self.eval_metrics_summ = self._eval_metrics_fn()

        # Fusion of all summaries
        self.merged = tf.summary.merge_all()

        # Tensorflow session for graph management
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Get handles for iterators
        with tf.variable_scope('handles'):
            handles_list = self.sess.run(
                [iterator.string_handle() for iterator in iterators_list])
            self.handle_train = handles_list[0]
            self.handle_eval = handles_list[1]

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

        # For normalization
        self.cwt_stats_dict = {}

        # Save the parameters used to define this model
        with open(os.path.join(self.logdir, 'params.json'), 'w') as outfile:
            json.dump(self.params, outfile)

    def _update_cwt_stats(self, x, sub_id):
        print('Updating cwt stats for ID %02d' % sub_id)
        cwt_list = []
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)
        for i in range(niters):
            start_index = i * self.params[pkeys.BATCH_SIZE]
            end_index = (i + 1) * self.params[pkeys.BATCH_SIZE]
            batch = x[start_index:end_index]
            cwt = self.sess.run(
                self.cwt_prebn,
                feed_dict={
                    self.feats: batch,
                    self.training_ph: False
                })
            cwt_list.append(cwt)
        cwt_list = np.concatenate(cwt_list, axis=0)

        cwt_mean = cwt_list.mean(axis=(0, 1))
        cwt_variance = cwt_list.var(axis=(0, 1))

        # Shape (n_scales, n_channels)
        self.cwt_stats_dict.update(
            {sub_id: (cwt_mean, cwt_variance)}
        )

    def predict_dataset(self, data_inference: FeederDataset, verbose=False):
        with_augmented_page = self.params[pkeys.PREDICT_WITH_AUGMENTED_PAGE]
        border_size = self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]
        x_val, _ = data_inference.get_data_for_prediction(
            border_size=border_size,
            predict_with_augmented_page=False,
            verbose=False)
        sub_id_list = data_inference.get_ids()
        for single_x, single_id in zip(x_val, sub_id_list):
            self._update_cwt_stats(single_x, single_id)

        x_val, _ = data_inference.get_data_for_prediction(
            border_size=border_size,
            predict_with_augmented_page=with_augmented_page,
            verbose=False)

        probabilies_list = self.predict_proba_with_list(
            x_val, sub_id_list,
            verbose=verbose, with_augmented_page=with_augmented_page)
        # Now create PredictedDataset object
        probabilities_dict = {}
        all_ids = data_inference.get_ids()
        for k, sub_id in enumerate(all_ids):
            this_proba = probabilies_list[k]
            # Transform to whole-night probability vector
            this_proba = pages2seq(
                this_proba,
                data_inference.get_subject_pages(sub_id, constants.WN_RECORD))
            probabilities_dict[sub_id] = this_proba
        prediction = PredictedDataset(
            data_inference,
            probabilities_dict,
            params=self.params)
        return prediction

    def predict_proba(self, x, sub_id, with_augmented_page=False):
        """Predicts the class probabilities over the data x."""
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)
        probabilities_list = []
        for i in range(niters):
            start_index = i*self.params[pkeys.BATCH_SIZE]
            end_index = (i+1)*self.params[pkeys.BATCH_SIZE]
            batch = x[start_index:end_index]

            if with_augmented_page:
                page_size = self.params[pkeys.PAGE_DURATION] * self.params[pkeys.FS]
                border_size = self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]
                input_size = page_size + 2 * border_size
                start_left = int(page_size / 4)
                end_left = int(start_left + input_size)
                start_right = int(3 * page_size / 4)
                end_right = int(start_right + input_size)

                batch_left = batch[:, start_left:end_left]
                batch_right = batch[:, start_right:end_right]

                batch_left_cwt = self.sess.run(
                    self.cwt_prebn,
                    feed_dict={self.feats: batch_left, self.training_ph: False})
                batch_left_cwt = self._normalize_cwt(batch_left_cwt, sub_id)
                proba_left = self.sess.run(
                    self.probabilities,
                    feed_dict={
                        self.cwt_prebn: batch_left_cwt,
                        self.training_ph: False
                    })

                batch_right_cwt = self.sess.run(
                    self.cwt_prebn,
                    feed_dict={self.feats: batch_right, self.training_ph: False})
                batch_right_cwt = self._normalize_cwt(batch_right_cwt, sub_id)
                proba_right = self.sess.run(
                    self.probabilities,
                    feed_dict={
                        self.cwt_prebn: batch_right_cwt,
                        self.training_ph: False
                    })
                # Keep central half of each
                length_out = proba_left.shape[1]
                start_crop = int(length_out / 4)
                end_crop = int(3 * length_out / 4)
                crop_left = proba_left[:, start_crop:end_crop, :]
                crop_right = proba_right[:, start_crop:end_crop, :]
                probabilities = np.concatenate([crop_left, crop_right], axis=1)
            else:
                batch_cwt = self.sess.run(
                    self.cwt_prebn,
                    feed_dict={self.feats: batch, self.training_ph: False})
                batch_cwt = self._normalize_cwt(batch_cwt, sub_id)
                probabilities = self.sess.run(
                    self.probabilities,
                    feed_dict={
                        self.cwt_prebn: batch_cwt,
                        self.training_ph: False
                    })
            probabilities_list.append(probabilities)
        final_probabilities = np.concatenate(probabilities_list, axis=0)

        # Keep only probability of class 1
        final_probabilities = final_probabilities[..., 1]

        # Transform to float16 precision
        final_probabilities = final_probabilities.astype(np.float16)

        return final_probabilities

    def predict_proba_with_list(
            self, x_list, sub_id_list, verbose=False, with_augmented_page=False):
        """Predicts the class probabilities over a list of data x."""
        probabilities_list = []
        for i, x in enumerate(x_list):
            if verbose:
                print('Predicting %d / %d ... '
                      % (i+1, len(x_list)), end='', flush=True)
            this_pred = self.predict_proba(
                x, sub_id_list[i], with_augmented_page=with_augmented_page)
            probabilities_list.append(this_pred)
            if verbose:
                print('Done', flush=True)
        return probabilities_list

    def _personalize_bn(self, x):
        cwt_list = []
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)
        for i in range(niters):
            start_index = i * self.params[pkeys.BATCH_SIZE]
            end_index = (i + 1) * self.params[pkeys.BATCH_SIZE]
            batch = x[start_index:end_index]
            cwt = self.sess.run(
                self.cwt_prebn,
                feed_dict={
                    self.feats: batch,
                    self.training_ph: False
                })
            cwt_list.append(cwt)
        cwt_list = np.concatenate(cwt_list, axis=0)
        # print(cwt_list.shape)
        n_channels = cwt_list.shape[-1]
        # print(n_channels)
        for k in range(n_channels):
            # Compute statistics
            new_mean = cwt_list[..., k].mean(axis=(0, 1))
            new_variance = cwt_list[..., k].var(axis=(0, 1))
            # print(new_mean.shape, new_variance.shape)
            # Select right variables
            my_vars = [var for var in self.ind_variables if 'bn_%d' % k in var.name]
            old_mean = [var for var in my_vars if 'mean' in var.name][0]
            old_variance = [var for var in my_vars if 'variance' in var.name][0]
            # print(old_mean, old_variance)
            # Update variables
            self.sess.run(tf.assign(old_mean, new_mean))
            self.sess.run(tf.assign(old_variance, new_variance))

    def _normalize_cwt(self, cwt, sub_ids):
        # print('Not normalizing cwt of shape', cwt.shape)
        # print(sub_ids)
        cwt = self._normalize_cwt_4real(cwt, sub_ids)
        return cwt

    def _normalize_cwt_4real(self, cwt, sub_ids):
        sub_ids = np.asarray(sub_ids)
        if sub_ids.ndim == 0:
            # This means that the same id applies to the whole batch
            sub_ids = np.stack([sub_ids] * cwt.shape[0])

        # Now for each example there is a sub id
        # Let's collect the necessary stats
        batch_mean = []
        batch_var = []
        for single_id in sub_ids:
            this_mean, this_var = self.cwt_stats_dict[single_id]
            # These arrays have shape [n_freq, n_channels]
            batch_mean.append(this_mean)
            batch_var.append(this_var)
        # Here we'll have shape [batch, n_freq, n_channels]
        batch_mean = np.stack(batch_mean, axis=0)
        batch_var = np.stack(batch_var, axis=0)
        # Now we add dummy dimension in time dimension
        batch_mean = np.expand_dims(batch_mean, 1)
        batch_var = np.expand_dims(batch_var, 1)
        # Now we have shape [batch, 1, n_freq, n_channels]
        # Broadcasting should be enough
        # print('Mean and var for batch', batch_mean.shape, batch_var.shape)
        cwt = (cwt - batch_mean) / np.sqrt(batch_var + 1e-3)
        return cwt

    def evaluate(self, x, y, sub_ids):
        """Evaluates the model, averaging evaluation metrics over batches."""
        self._init_iterator_eval(x, y, sub_ids)
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)
        metrics_list = []
        for i in range(niters):

            batch_cwt, batch_labels, batch_sub_ids = self.sess.run(
                [self.cwt_prebn, self.labels, self.sub_ids],
                feed_dict={self.training_ph: False,
                           self.handle_ph: self.handle_eval})

            batch_cwt = self._normalize_cwt(batch_cwt, batch_sub_ids)

            eval_metrics = self.sess.run(
                self.eval_metrics_dict,
                feed_dict={self.training_ph: False,
                           self.cwt_prebn: batch_cwt,
                           self.labels: batch_labels})

            metrics_list.append(eval_metrics)
        # Average
        mean_metrics = {}
        for key in self.eval_metrics_dict:
            value = 0
            for i in range(niters):
                value += metrics_list[i][key]
            mean_metrics[key] = value / niters
        # Create summary to write
        feed_dict = {}
        for key in self.eval_metrics_dict:
            feed_dict.update(
                {self.eval_metrics_dict[key]:  mean_metrics[key]}
            )
        mean_loss, mean_metrics, mean_summ = self.sess.run(
            [self.loss, self.batch_metrics_dict, self.eval_metrics_summ],
            feed_dict=feed_dict)
        return mean_loss, mean_metrics, mean_summ

    def load_checkpoint(self, ckptdir):
        """Loads variables from a checkpoint."""
        self.saver.restore(self.sess, ckptdir)

    def _init_iterator_train(
            self,
            x_train_1, y_train_1, sub_ids_train_1,
            x_train_2, y_train_2, sub_ids_train_2):
        """Init the train iterator."""
        self.sess.run(self.iterator_train.initializer,
                      feed_dict={self.feats_train_1_ph: x_train_1,
                                 self.labels_train_1_ph: y_train_1,
                                 self.sub_ids_train_1_ph: sub_ids_train_1,
                                 self.feats_train_2_ph: x_train_2,
                                 self.labels_train_2_ph: y_train_2,
                                 self.sub_ids_train_2_ph: sub_ids_train_2
                                 })

    def _init_iterator_eval(self, x_eval, y_eval, sub_ids_eval):
        """Init the evaluation iterator."""
        self.sess.run(self.iterator_eval.initializer,
                      feed_dict={self.feats_eval_ph: x_eval,
                                 self.labels_eval_ph: y_eval,
                                 self.sub_ids_eval_ph: sub_ids_eval
                                 })

    def _update_learning_rate(self, update_factor, ckptdir=None):
        # Restore checkpoint
        if ckptdir:
            self.load_checkpoint(ckptdir)
        # Reset optimizer variables (like moving averages)
        self.sess.run(self.reset_optimizer)
        # Decrease learning rate
        self.lr_updates = self.lr_updates + 1
        total_factor = update_factor ** self.lr_updates
        new_lr = self.params[pkeys.LEARNING_RATE] * total_factor
        self.sess.run(tf.assign(self.learning_rate, new_lr))
        return new_lr

    def _single_train_iteration(self):
        # print('Retrieving training batch')
        batch_cwt, batch_labels, batch_ids = self.sess.run(
            [self.cwt_prebn, self.labels, self.sub_ids],
            feed_dict={self.training_ph: True,
                       self.handle_ph: self.handle_train})

        batch_cwt = self._normalize_cwt(batch_cwt, batch_ids)

        # print('Reentering normalized batch')

        self.sess.run(
            self.train_step,
            feed_dict={self.training_ph: True,
                       self.cwt_prebn: batch_cwt,
                       self.labels: batch_labels})

    def _initialize_variables(self):
        self.sess.run(self.init_op)

    def fit(self, data_train, data_val):
        """This method has to be implemented."""
        pass

    def _train_map_fn(self, feat, label, sub_ids):
        """This method has to be implemented."""
        return feat, label, sub_ids

    def _eval_map_fn(self, feat, label, sub_ids):
        """This method has to be implemented."""
        return feat, label, sub_ids

    def _model_fn(self):
        """This method has to be implemented"""
        logits = None
        probabilities = None
        cwt_prebn = None
        return logits, probabilities, cwt_prebn

    def _loss_fn(self):
        """This method has to be implemented"""
        loss = None
        loss_summ = None
        return loss, loss_summ

    def _optimizer_fn(self):
        """This method has to be implemented"""
        train_step = None
        reset_optimizer_op = None
        grad_norm_summ = None
        return train_step, reset_optimizer_op, grad_norm_summ

    def _batch_metrics_fn(self):
        """This method has to be implemented"""
        metrics_dict = {}
        metrics_summ = None
        return metrics_dict, metrics_summ

    def _eval_metrics_fn(self):
        """This method has to be implemented"""
        metrics_dict = {}
        metrics_summ = None
        return metrics_dict, metrics_summ
