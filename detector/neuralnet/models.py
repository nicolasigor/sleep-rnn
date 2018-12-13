"""models.py: Module that defines trainable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from utils import param_keys
from utils import constants
from utils import errors
from .base_model import BaseModel
from .base_model import KEY_LOSS
from . import networks
from . import losses, optimizers, metrics

# Metrics dict
KEY_TP = 'tp'
KEY_FP = 'fp'
KEY_FN = 'fn'
KEY_PRECISION = 'precision'
KEY_RECALL = 'recall'
KEY_F1_SCORE = 'f1_score'


class WaveletBLSTM(BaseModel):
    """ Model that manages the implemented network."""

    def __init__(self, params, logdir='logs'):
        """Constructor.

        Feat and label shapes can be obtained from params for this model.
        """
        self.params = param_keys.default_params
        self.params.update(params)  # Overwrite defaults
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        augmented_input_length = 2*(page_size + border_size)
        feat_train_shape = [augmented_input_length]
        label_train_shape = feat_train_shape
        feat_eval_shape = [page_size + 2 * border_size]
        label_eval_shape = [page_size/8]
        super().__init__(
            feat_train_shape,
            label_train_shape,
            feat_eval_shape,
            label_eval_shape,
            params, logdir)

    def get_border_size(self):
        border_duration = self.params[param_keys.BORDER_DURATION]
        fs = self.params[param_keys.FS]
        border_size = fs * border_duration
        return border_size

    def get_page_size(self):
        page_duration = self.params[param_keys.PAGE_DURATION]
        fs = self.params[param_keys.FS]
        page_size = fs * page_duration
        return page_size

    def check_train_inputs(self, x_train, y_train, x_val, y_val):
        """Ensures that validation data has the proper shape."""
        time_stride = self.params[param_keys.TIME_RESOLUTION_FACTOR]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        crop_size = page_size + 2 * border_size
        if x_train.shape[1] == x_val.shape[1]:
            # If validation has augmented pages
            x_val = x_val[:, page_size // 2:-page_size // 2]
            y_val = y_val[:, page_size // 2:-page_size // 2]
        if y_val.shape[1] == crop_size:
            # We need to remove borders and downsampling for val labels.
            y_val = y_val[:, border_size:-border_size:time_stride]
        return x_train, y_train, x_val, y_val

    def fit(self, x_train, y_train, x_val, y_val):
        # TODO: decay learning rate and early stopping
        """Fits the model to the training data."""
        x_train, y_train, x_val, y_val = self.check_train_inputs(
            x_train, y_train, x_val, y_val)
        iter_per_epoch = x_train.shape[0] // self.params[param_keys.BATCH_SIZE]
        niters = self.params[param_keys.MAX_ITERATIONS]

        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Iters per epoch %d, '
              'Training examples %d, Total iterations %d' %
              (self.params[param_keys.BATCH_SIZE],
               iter_per_epoch,
               x_train.shape[0], niters))
        start_time = time.time()

        self._initialize_variables()
        self._init_iterator_train(x_train, y_train)

        # Training loop
        nstats = self.params[param_keys.NSTATS]
        for it in range(1, niters+1):
            self._single_train_iteration()
            if it % nstats == 0 or it == 1 or it == niters:
                # Report stuff
                # Training report is batch report
                train_loss, train_metrics, train_summ = self.sess.run(
                    [self.loss, self.batch_metrics_dict, self.merged],
                    feed_dict={self.training_ph: False,
                               self.handle_ph: self.handle_train})
                self.train_writer.add_summary(train_summ, it)
                # Validation report is entire set
                val_loss, val_metrics, val_summ = self.evaluate(x_val, y_val)
                self.val_writer.add_summary(val_summ, it)

                elapsed = time.time() - start_time
                loss_print = ('loss train %1.6f val %1.6f'
                              % (train_loss,
                                 val_loss))
                f1_print = ('f1 train %1.6f val %1.6f'
                              % (train_metrics[KEY_F1_SCORE],
                                 val_metrics[KEY_F1_SCORE]))
                print('It %6.0d/%d - %s - %s - E.T. %1.4f s'
                      % (it, niters, loss_print, f1_print, elapsed))
        # Save fitted model checkpoint
        save_path = self.saver.save(self.sess, self.ckptdir)
        print('Model saved at %s' % save_path)

    def _train_map_fn(self, feat, label):
        """Random cropping.

        This method is used to preprocess features and labels of single
        examples with a random cropping
        """
        time_stride = self.params[param_keys.TIME_RESOLUTION_FACTOR]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        crop_size = page_size + 2 * border_size
        # Random crop
        label_cast = tf.cast(label, dtype=tf.float32)
        stack = tf.stack([feat, label_cast], axis=0)
        stack_crop = tf.random_crop(stack, [2, crop_size])
        feat = stack_crop[0, :]
        # Throw borders for labels, skipping steps
        label_cast = stack_crop[1, border_size:-border_size:time_stride]
        label = tf.cast(label_cast, dtype=tf.int32)
        return feat, label

    def _model_fn(self):
        logits, probabilities = networks.wavelet_blstm_net(
            self.feats, self.params, self.training_ph)
        return logits, probabilities

    def _loss_fn(self):
        type_loss = self.params[param_keys.TYPE_LOSS]
        errors.check_valid_value(
            type_loss, 'type_loss',
            [constants.CROSS_ENTROPY_LOSS, constants.DICE_LOSS])

        if type_loss == constants.CROSS_ENTROPY_LOSS:
            loss, loss_summ = losses.cross_entropy_loss_fn(
                self.logits, self.labels, self.params[param_keys.CLASS_WEIGHTS])
        else:
            loss, loss_summ = losses.dice_loss_fn(
                self.probabilities[..., 1], self.labels)
        return loss,loss_summ

    def _optimizer_fn(self):
        type_optimizer = self.params[param_keys.TYPE_OPTIMIZER]
        errors.check_valid_value(
            type_optimizer, 'type_optimizer',
            [constants.ADAM_OPTIMIZER, constants.SGD_OPTIMIZER,
             constants.RMSPROP_OPTIMIZER])

        if type_optimizer == constants.ADAM_OPTIMIZER:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.adam_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[param_keys.CLIP_GRADIENTS],
                self.params[param_keys.CLIP_NORM])
        elif type_optimizer == constants.RMSPROP_OPTIMIZER:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.rmsprop_optimizer_fn(
                self.loss, self.learning_rate, self.params[param_keys.MOMENTUM],
                self.params[param_keys.CLIP_GRADIENTS],
                self.params[param_keys.CLIP_NORM])
        else:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.sgd_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[param_keys.MOMENTUM],
                self.params[param_keys.CLIP_GRADIENTS],
                self.params[param_keys.CLIP_NORM],
                self.params[param_keys.USE_NESTEROV_MOMENTUM])
        return train_step, reset_optimizer_op, grad_norm_summ

    def _batch_metrics_fn(self):
        with tf.variable_scope('batch_metrics'):
            tp, fp, fn = metrics.confusion_matrix(self.logits, self.labels)
            precision, recall, f1_score = metrics.precision_recall_f1score(
                tp, fp, fn)
            prec_summ = tf.summary.scalar(KEY_PRECISION, precision)
            rec_summ = tf.summary.scalar(KEY_RECALL, recall)
            f1_summ = tf.summary.scalar(KEY_F1_SCORE, f1_score)
            batch_metrics_dict = {
                KEY_PRECISION: precision,
                KEY_RECALL: recall,
                KEY_F1_SCORE: f1_score,
                KEY_TP: tp,
                KEY_FP: fp,
                KEY_FN: fn
            }
            batch_metrics_summ = tf.summary.merge(
                [prec_summ, rec_summ, f1_summ])
        return batch_metrics_dict, batch_metrics_summ

    def _eval_metrics_fn(self):
        with tf.variable_scope('eval_metrics'):
            eval_metrics_dict = {
                KEY_TP: self.batch_metrics_dict[KEY_TP],
                KEY_FP: self.batch_metrics_dict[KEY_FP],
                KEY_FN: self.batch_metrics_dict[KEY_FN],
                KEY_LOSS: self.loss
            }
            eval_metrics_summ = [
                self.loss_sum,
                self.batch_metrics_summ
            ]
            eval_metrics_summ = tf.summary.merge(eval_metrics_summ)
        return eval_metrics_dict, eval_metrics_summ
