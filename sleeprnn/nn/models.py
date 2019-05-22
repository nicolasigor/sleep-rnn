"""models.py: Module that defines trainable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import numpy as np
import tensorflow as tf

from sleeprnn.common import pkeys
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from .base_model import BaseModel
from .base_model import KEY_LOSS
from . import networks
from . import losses, optimizers, metrics, augmentations

# Metrics dict
KEY_TP = 'tp'
KEY_FP = 'fp'
KEY_FN = 'fn'
KEY_PRECISION = 'precision'
KEY_RECALL = 'recall'
KEY_F1_SCORE = 'f1_score'

# Fit dicts
KEY_ITER = 'iteration'


# TODO: Remove BaseModel class (is unnecessary)
class WaveletBLSTM(BaseModel):
    """ Model that manages the implemented network."""

    def __init__(self, params, logdir='logs'):
        """Constructor.

        Feat and label shapes can be obtained from params for this model.
        """
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        augmented_input_length = 2*(page_size + border_size)
        feat_train_shape = [augmented_input_length]
        label_train_shape = feat_train_shape
        feat_eval_shape = [page_size + 2 * border_size]
        label_eval_shape = [page_size / 8]
        super().__init__(
            feat_train_shape,
            label_train_shape,
            feat_eval_shape,
            label_eval_shape,
            params, logdir)

    def get_border_size(self):
        border_duration = self.params[pkeys.BORDER_DURATION]
        fs = self.params[pkeys.FS]
        border_size = fs * border_duration
        return border_size

    def get_page_size(self):
        page_duration = self.params[pkeys.PAGE_DURATION]
        fs = self.params[pkeys.FS]
        page_size = fs * page_duration
        return page_size

    def check_train_inputs(self, x_train, y_train, x_val, y_val):
        """Ensures that validation data has the proper shape."""
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
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

    def fit(
            self,
            data_train: FeederDataset,
            data_val: FeederDataset,
            verbose=False):
        """Fits the model to the training data."""
        border_size = self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]
        x_train, y_train = data_train.get_data_for_training(
            border_size=border_size,
            verbose=verbose)
        x_val, y_val = data_val.get_data_for_training(
            border_size=border_size,
            verbose=verbose)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        # Shuffle training set
        x_train, y_train = utils.shuffle_data(
            x_train, y_train, seed=0)

        print('Training set shape', x_train.shape, y_train.shape)
        print('Validation set shape', x_val.shape, y_val.shape)

        x_train, y_train, x_val, y_val = self.check_train_inputs(
            x_train, y_train, x_val, y_val)
        iter_per_epoch = x_train.shape[0] // self.params[pkeys.BATCH_SIZE]
        niters = self.params[pkeys.MAX_ITERS]

        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Iters per epoch %d, '
              'Training examples %d, Max iterations %d' %
              (self.params[pkeys.BATCH_SIZE],
               iter_per_epoch,
               x_train.shape[0], niters))
        print('Initial learning rate:', self.params[pkeys.LEARNING_RATE])
        start_time = time.time()

        # split set into two parts
        x_train_1, y_train_1, x_train_2, y_train_2 = self._split_train(
            x_train, y_train)

        self._initialize_variables()
        self._init_iterator_train(x_train_1, y_train_1, x_train_2, y_train_2)

        # Improvement criterion
        model_criterion = {
            KEY_ITER: 0,
            KEY_LOSS: 1e10,
            KEY_F1_SCORE: 0
        }
        rel_tol_criterion = self.params[pkeys.REL_TOL_CRITERION]
        iter_last_lr_update = 0

        if self.params[pkeys.MAX_LR_UPDATES] is None:
            self.params[pkeys.MAX_LR_UPDATES] = 1e15

        lr_update_criterion = self.params[pkeys.LR_UPDATE_CRITERION]
        checks.check_valid_value(
            lr_update_criterion,
            'lr_update_criterion',
            [constants.LOSS_CRITERION, constants.METRIC_CRITERION])

        # Training loop
        nstats = self.params[pkeys.ITERS_STATS]
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

                if lr_update_criterion == constants.LOSS_CRITERION:
                    improvement_criterion = val_loss < (1.0 - rel_tol_criterion) * model_criterion[KEY_LOSS]
                else:
                    improvement_criterion = val_metrics[KEY_F1_SCORE] > (1.0 + rel_tol_criterion) * model_criterion[KEY_F1_SCORE]

                if improvement_criterion:
                    # Update last time the improvement criterion was met
                    model_criterion[KEY_LOSS] = val_loss
                    model_criterion[KEY_F1_SCORE] = val_metrics[KEY_F1_SCORE]
                    model_criterion[KEY_ITER] = it

                # Check LR update criterion

                # The model has not improved enough
                lr_criterion_1 = (it - model_criterion[KEY_ITER]) >= self.params[pkeys.ITERS_LR_UPDATE]
                # The last lr update is far enough
                lr_criterion_2 = (it - iter_last_lr_update) >= self.params[pkeys.ITERS_LR_UPDATE]
                lr_criterion = lr_criterion_1 and lr_criterion_2
                if lr_criterion:
                    if self.lr_updates < self.params[pkeys.MAX_LR_UPDATES]:
                        new_lr = self._update_learning_rate(
                            self.params[pkeys.LR_UPDATE_FACTOR])
                        print('    Learning rate update (%d). New value: %s'
                              % (self.lr_updates, new_lr))
                        iter_last_lr_update = it
                    else:
                        print('    Maximum number (%d) of learning rate '
                              'updates reached. Stopping training.'
                              % self.params[pkeys.MAX_LR_UPDATES])
                        # Since we stop training, redefine number of iters
                        niters = it
                        break

        val_loss, val_metrics, _ = self.evaluate(x_val, y_val)
        last_model = {
            KEY_ITER: niters,
            KEY_LOSS: float(val_loss),
            KEY_F1_SCORE: float(val_metrics[KEY_F1_SCORE])
        }

        # Final stats
        elapsed = time.time() - start_time
        print('\n\nTotal training time: %1.4f s' % elapsed)
        print('Ending at iteration %d' % last_model[KEY_ITER])
        print('Validation loss %1.6f - f1 %1.6f'
              % (last_model[KEY_LOSS], last_model[KEY_F1_SCORE]))

        save_path = self.saver.save(self.sess, self.ckptdir)
        print('Model saved at %s' % save_path)

        # Save last model quick info
        with open(os.path.join(self.logdir, 'last_model.json'), 'w') as outfile:
            json.dump(last_model, outfile)

    def _train_map_fn(self, feat, label):
        """Random cropping.

        This method is used to preprocess features and labels of single
        examples with a random cropping
        """
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
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
        # Apply data augmentation
        # feat, label = self._augmentation_fn(feat, label)
        return feat, label

    def _augmentation_fn(self, feat, label):
        rescale_proba = self.params[pkeys.AUG_RESCALE_NORMAL_PROBA]
        rescale_std = self.params[pkeys.AUG_RESCALE_NORMAL_STD]
        noise_proba = self.params[pkeys.AUG_GAUSSIAN_NOISE_PROBA]
        noise_std = self.params[pkeys.AUG_GAUSSIAN_NOISE_STD]

        rescale_unif_proba = self.params[pkeys.AUG_RESCALE_UNIFORM_PROBA]
        rescale_unif_intens = self.params[pkeys.AUG_RESCALE_UNIFORM_INTENSITY]

        print('rescale proba, std:', rescale_proba, rescale_std)
        print('noise proba, std:', noise_proba, noise_std)
        print('rescale unif proba, intens:', rescale_unif_proba, rescale_unif_intens)

        if rescale_proba > 0:
            feat = augmentations.rescale_normal(
                feat, rescale_proba, rescale_std)
        if noise_proba > 0:
            feat = augmentations.gaussian_noise(
                feat, noise_proba, noise_std)

        if rescale_unif_proba > 0:
            feat = augmentations.rescale_uniform(
                feat, rescale_unif_proba, rescale_unif_intens)

        return feat, label

    def _model_fn(self):
        model_version = self.params[pkeys.MODEL_VERSION]
        checks.check_valid_value(
            model_version, 'model_version',
            [
                constants.DUMMY,
                constants.V1,
                constants.V4
             ])
        if model_version == constants.V1:
            model_fn = networks.wavelet_blstm_net_v1
        elif model_version == constants.V4:
            model_fn = networks.wavelet_blstm_net_v4
        else:
            model_fn = networks.dummy_net

        logits, probabilities, cwt_prebn = model_fn(
            self.feats, self.params, self.training_ph)
        return logits, probabilities, cwt_prebn

    def _loss_fn(self):
        type_loss = self.params[pkeys.TYPE_LOSS]
        checks.check_valid_value(
            type_loss, 'type_loss',
            [constants.CROSS_ENTROPY_LOSS, constants.DICE_LOSS])

        if type_loss == constants.CROSS_ENTROPY_LOSS:
            loss, loss_summ = losses.cross_entropy_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS])
        else:
            loss, loss_summ = losses.dice_loss_fn(
                self.probabilities[..., 1], self.labels)
        return loss,loss_summ

    def _optimizer_fn(self):
        type_optimizer = self.params[pkeys.TYPE_OPTIMIZER]
        checks.check_valid_value(
            type_optimizer, 'type_optimizer',
            [constants.ADAM_OPTIMIZER, constants.SGD_OPTIMIZER,
             constants.RMSPROP_OPTIMIZER])

        if type_optimizer == constants.ADAM_OPTIMIZER:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.adam_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[pkeys.CLIP_NORM])
        elif type_optimizer == constants.RMSPROP_OPTIMIZER:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.rmsprop_optimizer_fn(
                self.loss, self.learning_rate, self.params[pkeys.MOMENTUM],
                self.params[pkeys.CLIP_NORM])
        else:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.sgd_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[pkeys.MOMENTUM],
                self.params[pkeys.CLIP_NORM],
                self.params[pkeys.USE_NESTEROV_MOMENTUM])
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

    def _split_train(self, x_train, y_train):
        n_train = x_train.shape[0]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        # Remove to recover single page from augmented page
        remove_size = border_size + page_size // 2
        activity = y_train[:, remove_size:-remove_size]
        activity = np.sum(activity, axis=1)

        # Find pages with activity
        exists_activity_idx = np.where(activity > 0)[0]

        n_with_activity = exists_activity_idx.shape[0]

        print('Pages with activity: %d (%1.2f %% of total)'
              % (n_with_activity, 100 * n_with_activity / n_train))

        if n_with_activity < n_train/2:
            print('Balancing strategy: zero/exists activity')
            zero_activity_idx = np.where(activity == 0)[0]
            # Pages without any activity
            x_train_1 = x_train[zero_activity_idx]
            y_train_1 = y_train[zero_activity_idx]
            # Pages with activity
            x_train_2 = x_train[exists_activity_idx]
            y_train_2 = y_train[exists_activity_idx]
            print('Pages without activity:', x_train_1.shape)
            print('Pages with activity:', x_train_2.shape)
        else:
            print('Balancing strategy: low/high activity')
            sorted_idx = np.argsort(activity)
            low_activity_idx = sorted_idx[:int(n_train/2)]
            high_activity_idx = sorted_idx[int(n_train/2):]
            # Pages with low activity
            x_train_1 = x_train[low_activity_idx]
            y_train_1 = y_train[low_activity_idx]
            # Pages with high activity
            x_train_2 = x_train[high_activity_idx]
            y_train_2 = y_train[high_activity_idx]
            print('Pages with low activity:', x_train_1.shape)
            print('Pages with high activity:', x_train_2.shape)

        # # Split into pages with activity, and pages without any
        # # zero_activity_idx = np.where(activity == 0)[0]
        # # exists_activity_idx = np.where(activity > 0)[0]
        #
        # # # Sorting in ascending order (low to high)
        # sorted_idx = np.argsort(activity)
        # low_activity_idx = sorted_idx[:int(n_train/2)]
        # high_activity_idx = sorted_idx[int(n_train/2):]
        #
        # # Pages without any activity
        # # x_train_1 = x_train[zero_activity_idx]
        # # y_train_1 = y_train[zero_activity_idx]
        #
        # # Pages with activity
        # # x_train_2 = x_train[exists_activity_idx]
        # # y_train_2 = y_train[exists_activity_idx]
        #
        # # print('Pages without activity:', x_train_1.shape)
        # # print('Pages with activity:', x_train_2.shape)
        #
        # # Pages with low activity
        # x_train_1 = x_train[low_activity_idx]
        # y_train_1 = y_train[low_activity_idx]
        #
        # # Pages with high activity
        # x_train_2 = x_train[high_activity_idx]
        # y_train_2 = y_train[high_activity_idx]
        #
        # print('Pages with low activity:', x_train_1.shape)
        # print('Pages with high activity:', x_train_2.shape)

        return x_train_1, y_train_1, x_train_2, y_train_2
