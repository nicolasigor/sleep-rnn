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
from sleeprnn.detection import threshold_optimization
from sleeprnn.detection.metrics import matching_with_list, metric_vs_iou_macro_average, metric_vs_iou_micro_average
from sleeprnn.detection.feeder_dataset import FeederDataset
from .base_model import BaseModel
from .base_model import KEY_LOSS
from . import networks, networks_v2
from . import losses, optimizers, metrics, augmentations

# Metrics dict
KEY_TP = 'tp'
KEY_FP = 'fp'
KEY_FN = 'fn'
KEY_PRECISION = 'precision'
KEY_RECALL = 'recall'
KEY_F1_SCORE = 'f1_score'
KEY_AF1 = 'af1'

# Fit dicts
KEY_ITER = 'iteration'


# TODO: Remove BaseModel class (is unnecessary)
class WaveletBLSTM(BaseModel):
    """ Model that manages the implemented network."""

    def __init__(self, params=None, logdir='logs'):
        """Constructor.

        Feat and label shapes can be obtained from params for this model.
        """
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        augmented_input_length = 2*(page_size + border_size)
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        feat_train_shape = [augmented_input_length]
        label_train_shape = feat_train_shape
        feat_eval_shape = [page_size + 2 * border_size]
        label_eval_shape = [page_size / time_stride]
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

    def check_train_inputs(self, x_train, y_train, m_train, x_val, y_val, m_val):
        """Ensures that validation data has the proper shape."""
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        crop_size = page_size + 2 * border_size
        if x_train.shape[1] == x_val.shape[1]:
            # If validation has augmented pages
            x_val = x_val[:, page_size // 2:-page_size // 2]
            y_val = y_val[:, page_size // 2:-page_size // 2]
            m_val = m_val[:, page_size // 2:-page_size // 2]
        if y_val.shape[1] == crop_size:
            # We need to remove borders and downsampling for val labels.
            y_val = y_val[:, border_size:-border_size]
            m_val = m_val[:, border_size:-border_size]
            aligned_down = self.params[pkeys.ALIGNED_DOWNSAMPLING]
            if aligned_down:
                print('ALIGNED DOWNSAMPLING at checking inputs for fit')
                y_val_dtype = y_val.dtype
                y_val = y_val.reshape((-1, int(page_size/time_stride), time_stride))
                y_val = np.round(y_val.mean(axis=-1) + 1e-3).astype(y_val_dtype)

                m_val_dtype = m_val.dtype
                m_val = m_val.reshape((-1, int(page_size / time_stride), time_stride))
                m_val = np.round(m_val.mean(axis=-1) + 1e-3).astype(m_val_dtype)
            else:
                y_val = y_val[:, ::time_stride]
                m_val = m_val[:, ::time_stride]
        return x_train, y_train, m_train, x_val, y_val, m_val

    def fit_without_validation(
            self,
            data_train: FeederDataset,
            fine_tune=False,
            extra_data_train=None,
            verbose=False):
        """Fits the model to the training data."""
        border_size = int(self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS])
        forced_mark_separation_size = int(
            self.params[pkeys.FORCED_SEPARATION_DURATION] * self.params[pkeys.FS])

        x_train, y_train, m_train = data_train.get_data_for_training(
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            return_page_mask=True,
            verbose=verbose)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        m_train = np.concatenate(m_train, axis=0)

        # Add extra training data
        if extra_data_train is not None:
            x_extra, y_extra, m_extra = extra_data_train
            print('CHECK: Sum extra y:', y_extra.sum())
            print('Current train data x, y, m:', x_train.shape, y_train.shape, m_train.shape)
            x_train = np.concatenate([x_train, x_extra], axis=0)
            y_train = np.concatenate([y_train, y_extra], axis=0)
            m_train = np.concatenate([m_train, m_extra], axis=0)
            print('Extra data to be added x, y, m:', x_extra.shape, y_extra.shape, m_extra.shape)
            print('New train data', x_train.shape, y_train.shape, m_train.shape)

        # Shuffle training set
        list_of_outputs = utils.shuffle_data_collection([x_train, y_train, m_train], seed=0)
        x_train, y_train, m_train = list_of_outputs[0], list_of_outputs[1], list_of_outputs[2]

        print('Training set shape', x_train.shape, y_train.shape, m_train.shape)
        print('Validation set does not exist')

        x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2 = self._split_train(
            x_train, y_train, m_train)

        batch_size = self.params[pkeys.BATCH_SIZE]
        iters_resolution = 10
        n_smallest = min(x_train_1.shape[0], x_train_2.shape[0])
        iter_per_epoch = int(n_smallest / (batch_size / 2))
        iter_per_epoch = int((iter_per_epoch // iters_resolution) * iters_resolution)

        nstats = iter_per_epoch // self.params[pkeys.STATS_PER_EPOCH]
        niters_init = self.params[pkeys.PRETRAIN_EPOCHS_INIT] * iter_per_epoch
        niters_anneal = self.params[pkeys.PRETRAIN_EPOCHS_ANNEAL] * iter_per_epoch
        n_lr_updates = self.params[pkeys.PRETRAIN_MAX_LR_UPDATES]
        total_iters = niters_init + n_lr_updates * niters_anneal

        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Iters per epoch %d, '
              'Training examples %d, Init iters %d, Annealing iters %d, Total iters %d' %
              (self.params[pkeys.BATCH_SIZE],
               iter_per_epoch,
               x_train.shape[0], niters_init, niters_anneal, total_iters))
        print('Initial learning rate:', self.params[pkeys.LEARNING_RATE])
        print('Initial weight decay:', self.params[pkeys.WEIGHT_DECAY_FACTOR])

        if fine_tune:
            init_lr = self.params[pkeys.LEARNING_RATE]
            factor_fine_tune = self.params[pkeys.FACTOR_INIT_LR_FINE_TUNE]
            init_lr_fine_tune = init_lr * factor_fine_tune
            self.sess.run(self.reset_optimizer)
            self.sess.run(tf.assign(self.learning_rate, init_lr_fine_tune))
            print('Fine tuning with lr %s' % init_lr_fine_tune)
        else:
            self._initialize_variables()

        self._init_iterator_train(x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2)

        # Training loop
        start_time = time.time()
        last_elapsed = 0
        last_it = 0
        iter_last_lr_update = niters_init - niters_anneal

        for it in range(1, total_iters+1):
            self._single_train_iteration()
            if it % nstats == 0 or it == 1 or it == total_iters:
                # Report stuff. Training report is batch report
                train_loss, train_metrics, train_summ = self.sess.run(
                    [self.loss, self.batch_metrics_dict, self.merged],
                    feed_dict={self.training_ph: False, self.handle_ph: self.handle_train})
                self.train_writer.add_summary(train_summ, it)
                elapsed = time.time() - start_time
                time_rate_per_100 = 100 * (elapsed - last_elapsed) / (it - last_it)
                last_it = it
                last_elapsed = elapsed
                loss_print = ('loss train %1.4f' % train_loss)
                f1_print = ('f1 train %1.4f' % train_metrics[KEY_F1_SCORE])
                print('It %6.0d/%d - %s - %s - E.T. %1.2fs (%1.2fs/100it)'
                      % (it, total_iters, loss_print, f1_print, elapsed, time_rate_per_100))
            # The last lr update is far enough
            lr_criterion = (it - iter_last_lr_update) >= niters_anneal
            if lr_criterion:
                new_lr = self._update_learning_rate(self.params[pkeys.LR_UPDATE_FACTOR])
                print('    Learning rate update (%d). New value: %s' % (self.lr_updates, new_lr))
                iter_last_lr_update = it
        # Final stats
        iter_saved_model = total_iters
        elapsed = time.time() - start_time
        print('\n\nTotal training time: %1.4f s' % elapsed)
        print('Ending at iteration %d' % iter_saved_model)
        save_path = self.saver.save(self.sess, self.ckptdir)
        print('Model saved at %s' % save_path)
        last_model = {
            KEY_ITER: iter_saved_model,
            KEY_LOSS: 0,
            KEY_F1_SCORE: 0
        }
        # Save last model quick info
        with open(os.path.join(self.logdir, 'last_model.json'), 'w') as outfile:
            json.dump(last_model, outfile)

    def fit(
            self,
            data_train: FeederDataset,
            data_val: FeederDataset,
            fine_tune=False,
            extra_data_train=None,
            verbose=False):
        """Fits the model to the training data."""
        border_size = int(
            self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS])
        forced_mark_separation_size = int(
            self.params[pkeys.FORCED_SEPARATION_DURATION] * self.params[pkeys.FS])

        x_train, y_train, m_train = data_train.get_data_for_training(
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            return_page_mask=True,
            verbose=verbose)
        x_val, y_val, m_val = data_val.get_data_for_training(
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            return_page_mask=True,
            verbose=verbose)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        m_train = np.concatenate(m_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        m_val = np.concatenate(m_val, axis=0)

        # Add extra training data
        if extra_data_train is not None:
            x_extra, y_extra, m_extra = extra_data_train
            print('CHECK: Sum extra y:', y_extra.sum())
            print('Current train data x, y, m:', x_train.shape, y_train.shape, m_train.shape)
            x_train = np.concatenate([x_train, x_extra], axis=0)
            y_train = np.concatenate([y_train, y_extra], axis=0)
            m_train = np.concatenate([m_train, m_extra], axis=0)
            print('Extra data to be added x, y, m:', x_extra.shape, y_extra.shape, m_extra.shape)
            print('New train data', x_train.shape, y_train.shape, m_train.shape)

        # Shuffle training set
        list_of_outputs = utils.shuffle_data_collection([x_train, y_train, m_train], seed=0)
        x_train, y_train, m_train = list_of_outputs[0], list_of_outputs[1], list_of_outputs[2]

        print('Training set shape', x_train.shape, y_train.shape, m_train.shape)
        print('Validation set shape', x_val.shape, y_val.shape, m_val.shape)

        x_train, y_train, m_train, x_val, y_val, m_val = self.check_train_inputs(
            x_train, y_train, m_train, x_val, y_val, m_val)
        x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2 = self._split_train(
            x_train, y_train, m_train)

        batch_size = self.params[pkeys.BATCH_SIZE]
        iters_resolution = 10
        n_smallest = min(x_train_1.shape[0], x_train_2.shape[0])
        iter_per_epoch = int(n_smallest / (batch_size / 2))
        iter_per_epoch = int((iter_per_epoch // iters_resolution) * iters_resolution)

        niters = self.params[pkeys.MAX_EPOCHS] * iter_per_epoch
        iters_lr_update = self.params[pkeys.EPOCHS_LR_UPDATE] * iter_per_epoch
        nstats = iter_per_epoch // self.params[pkeys.STATS_PER_EPOCH]

        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Iters per epoch %d, '
              'Training examples %d, Max iterations %d' %
              (self.params[pkeys.BATCH_SIZE],
               iter_per_epoch,
               x_train.shape[0], niters))
        print('Initial learning rate:', self.params[pkeys.LEARNING_RATE])
        print('Initial weight decay:', self.params[pkeys.WEIGHT_DECAY_FACTOR])

        if fine_tune:
            init_lr = self.params[pkeys.LEARNING_RATE]
            factor_fine_tune = self.params[pkeys.FACTOR_INIT_LR_FINE_TUNE]
            init_lr_fine_tune = init_lr * factor_fine_tune
            self.sess.run(self.reset_optimizer)
            self.sess.run(tf.assign(self.learning_rate, init_lr_fine_tune))
            print('Fine tuning with lr %s' % init_lr_fine_tune)
        else:
            self._initialize_variables()

        self._init_iterator_train(x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2)

        # Improvement criterion
        model_criterion = {
            KEY_ITER: 0,
            KEY_LOSS: 1e10,
            KEY_F1_SCORE: 0,
            KEY_AF1: 0,
        }
        rel_tol_criterion = self.params[pkeys.REL_TOL_CRITERION]
        iter_last_lr_update = 0

        lr_update_criterion = self.params[pkeys.LR_UPDATE_CRITERION]
        checks.check_valid_value(
            lr_update_criterion,
            'lr_update_criterion',
            [constants.LOSS_CRITERION, constants.METRIC_CRITERION])
        print("Learning rate decay criterion: %s" % lr_update_criterion)

        # Validation events for AF1
        val_thr_space = {'min': 0.2, 'max': 0.8, 'step': 0.02}
        if 'moda' in data_val.dataset_name:
            val_avg_mode = constants.MICRO_AVERAGE
        else:
            val_avg_mode = constants.MACRO_AVERAGE
        print("Validation AF1 computed using %s and thr space %s:%s:%s" % (
            val_avg_mode, val_thr_space['min'], val_thr_space['step'], val_thr_space['max']))

        # Training loop
        start_time = time.time()
        last_elapsed = 0
        last_it = 0
        for it in range(1, niters+1):
            self._single_train_iteration()
            if it % nstats == 0 or it == 1 or it == niters:
                metric_msg = 'It %6.0d/%d' % (it, niters)
                # Train set report (mini-batch)
                train_loss, train_metrics, train_summ = self.sess.run(
                    [self.loss, self.batch_metrics_dict, self.merged],
                    feed_dict={self.training_ph: False, self.handle_ph: self.handle_train})
                self.train_writer.add_summary(train_summ, it)
                metric_msg += ' - train loss %1.4f f1 %1.4f' % (train_loss, train_metrics[KEY_F1_SCORE])
                if it % iter_per_epoch == 0 or it == 1 or it == niters:
                    # Val set report (whole set)
                    val_loss, val_metrics, val_summ = self.evaluate(x_val, y_val, m_val)
                    self.val_writer.add_summary(val_summ, it)
                    byevent_val_metrics, byevent_val_summ = self.evaluate_byevent(
                        data_val, val_thr_space, val_avg_mode)
                    self.val_writer.add_summary(byevent_val_summ, it)

                    metric_msg += ' - val loss %1.4f f1 %1.4f AF1 %1.4f (thr %1.2f)' % (
                        val_loss, val_metrics[KEY_F1_SCORE],
                        byevent_val_metrics['af1'], byevent_val_metrics['threshold'])
                    # Time passed
                    elapsed = time.time() - start_time
                    time_rate_per_100 = 100 * (elapsed - last_elapsed) / (it - last_it)
                    last_it = it
                    last_elapsed = elapsed
                    metric_msg += ' - E.T. %1.2fs (%1.2fs/100it)' % (elapsed, time_rate_per_100)
                    print(metric_msg)

                    if lr_update_criterion == constants.LOSS_CRITERION:
                        improvement_criterion = val_loss < (1.0 - rel_tol_criterion) * model_criterion[KEY_LOSS]
                    else:
                        improvement_criterion = byevent_val_metrics['af1'] > (1.0 + rel_tol_criterion) * model_criterion[KEY_AF1]
                    if improvement_criterion:
                        # Update last time the improvement criterion was met
                        model_criterion[KEY_LOSS] = val_loss
                        model_criterion[KEY_F1_SCORE] = val_metrics[KEY_F1_SCORE]
                        model_criterion[KEY_ITER] = it
                        model_criterion[KEY_AF1] = byevent_val_metrics['af1']
                        # Save best model
                        if self.params[pkeys.KEEP_BEST_VALIDATION]:
                            print("Checkpointing best model so far.")
                            self.saver.save(self.sess, self.ckptdir)

                    # Check LR update criterion

                    # The model has not improved for long time
                    lr_criterion_1 = (it - model_criterion[KEY_ITER]) >= iters_lr_update
                    # The last lr update is far enough
                    lr_criterion_2 = (it - iter_last_lr_update) >= iters_lr_update
                    lr_criterion = lr_criterion_1 and lr_criterion_2
                    if lr_criterion:
                        if self.lr_updates < self.params[pkeys.MAX_LR_UPDATES]:
                            # if self.params[pkeys.KEEP_BEST_VALIDATION]:
                            #     print('Restoring best model before lr update')
                            #     self.load_checkpoint(self.ckptdir)
                            new_lr = self._update_learning_rate(self.params[pkeys.LR_UPDATE_FACTOR])
                            print('    Learning rate update (%d). New value: %s' % (self.lr_updates, new_lr))
                            iter_last_lr_update = it
                        else:
                            print('    Maximum number (%d) of learning rate '
                                  'updates reached. Stopping training.'
                                  % self.params[pkeys.MAX_LR_UPDATES])
                            # Since we stop training, redefine number of iters
                            niters = it
                            break
                else:
                    print(metric_msg)

        if self.params[pkeys.KEEP_BEST_VALIDATION]:
            iter_saved_model = model_criterion[KEY_ITER]
            print('Restoring best model from it %d' % iter_saved_model)
            self.load_checkpoint(self.ckptdir)
        else:
            print('Keeping model from last iteration')
            iter_saved_model = niters

        val_loss, val_metrics, _ = self.evaluate(x_val, y_val, m_val)
        last_model = {
            KEY_ITER: iter_saved_model,
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

    def evaluate_byevent(
            self, validation_dataset, threshold_space, average_mode, iou_threshold_report=0.2):

        metric_vs_iou_fn_dict = {
            constants.MACRO_AVERAGE: metric_vs_iou_macro_average,
            constants.MICRO_AVERAGE: metric_vs_iou_micro_average}

        prediction_val = self.predict_dataset(validation_dataset, verbose=False)

        byevent_thr, byevent_af1 = threshold_optimization.fit_threshold(
            [validation_dataset], [prediction_val], threshold_space, average_mode, return_best_af1=True)
        prediction_val.set_probability_threshold(byevent_thr)

        val_events_list = validation_dataset.get_stamps()
        val_detections_list = prediction_val.get_stamps()

        iou_matching_list, _ = matching_with_list(val_events_list, val_detections_list)

        byevent_f1 = metric_vs_iou_fn_dict[average_mode](
            val_events_list, val_detections_list, [iou_threshold_report],
            metric_name=constants.F1_SCORE,
            iou_matching_list=iou_matching_list)[0]

        byevent_precision = metric_vs_iou_fn_dict[average_mode](
            val_events_list, val_detections_list, [iou_threshold_report],
            metric_name=constants.PRECISION,
            iou_matching_list=iou_matching_list)[0]

        byevent_recall = metric_vs_iou_fn_dict[average_mode](
            val_events_list, val_detections_list, [iou_threshold_report],
            metric_name=constants.RECALL,
            iou_matching_list=iou_matching_list)[0]

        nonzero_iou_list = [iou_matching[iou_matching > 0] for iou_matching in iou_matching_list]
        if average_mode == constants.MACRO_AVERAGE:
            miou_list = [np.mean(nonzero_iou) for nonzero_iou in nonzero_iou_list]
            byevent_miou = np.mean(miou_list)
        else:
            byevent_miou = np.concatenate(nonzero_iou_list).mean()

        byevent_metrics = {
            'threshold': byevent_thr,
            'af1': byevent_af1,
            'f1': byevent_f1,
            'recall': byevent_recall,
            'precision': byevent_precision,
            'miou': byevent_miou}

        byevent_summ = self.sess.run(
            self.byevent_metrics_summ, feed_dict={
                self.eval_threshold: byevent_thr,
                self.eval_af1: byevent_af1,
                self.eval_f1: byevent_f1,
                self.eval_precision: byevent_precision,
                self.eval_recall: byevent_recall,
                self.eval_miou: byevent_miou
            }
        )
        return byevent_metrics, byevent_summ

    def _eval_map_fn(self, feat, label, mask):
        label = tf.cast(label, tf.int32)
        mask = tf.cast(mask, tf.int32)
        return feat, label, mask

    def _train_map_fn(self, feat, label, mask):
        """Random cropping.

        This method is used to preprocess features and labels of single
        examples with a random cropping
        """

        # Prepare for training
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        crop_size = page_size + 2 * border_size
        # Random crop
        label_cast = tf.cast(label, dtype=tf.float32)
        mask_cast = tf.cast(mask, dtype=tf.float32)
        stack = tf.stack([feat, label_cast, mask_cast], axis=0)
        stack_crop = tf.random_crop(stack, [3, crop_size])
        feat = stack_crop[0, :]
        label = tf.cast(stack_crop[1, :], dtype=tf.int32)
        mask = tf.cast(stack_crop[2, :], dtype=tf.int32)

        # Apply data augmentation
        feat, label, mask = self._augmentation_fn(feat, label, mask)

        # Throw borders for labels, skipping steps
        # We need to remove borders and downsampling for val labels.
        label = label[border_size:-border_size]
        mask = mask[border_size:-border_size]
        aligned_down = self.params[pkeys.ALIGNED_DOWNSAMPLING]
        if aligned_down:
            print('ALIGNED DOWNSAMPLING at iterator')
            # Label downsampling
            label = tf.cast(label, tf.float32)
            label = tf.reshape(label, [-1, time_stride])
            label = tf.reduce_mean(label, axis=-1)
            label = tf.round(label + 1e-3)
            label = tf.cast(label, tf.int32)
            # Mask downsampling
            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, [-1, time_stride])
            mask = tf.reduce_mean(mask, axis=-1)
            mask = tf.round(mask + 1e-3)
            mask = tf.cast(mask, tf.int32)
        else:
            label = label[::time_stride]
            mask = mask[::time_stride]
        return feat, label, mask

    def _augmentation_fn(self, feat, label, mask):
        rescale_proba = self.params[pkeys.AUG_RESCALE_NORMAL_PROBA]
        rescale_std = self.params[pkeys.AUG_RESCALE_NORMAL_STD]
        noise_proba = self.params[pkeys.AUG_GAUSSIAN_NOISE_PROBA]
        noise_std = self.params[pkeys.AUG_GAUSSIAN_NOISE_STD]

        rescale_unif_proba = self.params[pkeys.AUG_RESCALE_UNIFORM_PROBA]
        rescale_unif_intens = self.params[pkeys.AUG_RESCALE_UNIFORM_INTENSITY]

        elastic_proba = self.params[pkeys.AUG_ELASTIC_PROBA]
        elastic_alpha = self.params[pkeys.AUG_ELASTIC_ALPHA]
        elastic_sigma = self.params[pkeys.AUG_ELASTIC_SIGMA]

        indep_noise_proba = self.params[pkeys.AUG_INDEP_GAUSSIAN_NOISE_PROBA]
        indep_noise_std = self.params[pkeys.AUG_INDEP_GAUSSIAN_NOISE_STD]

        random_waves_proba = self.params[pkeys.AUG_RANDOM_WAVES_PROBA]
        random_waves_params = self.params[pkeys.AUG_RANDOM_WAVES_PARAMS]

        random_anti_waves_proba = self.params[pkeys.AUG_RANDOM_ANTI_WAVES_PROBA]
        random_anti_waves_params = self.params[pkeys.AUG_RANDOM_ANTI_WAVES_PARAMS]

        false_spindles_single_cont_proba = self.params[pkeys.AUG_FALSE_SPINDLES_SINGLE_CONT_PROBA]
        false_spindles_single_cont_params = self.params[pkeys.AUG_FALSE_SPINDLES_SINGLE_CONT_PARAMS]

        print('rescale proba %s, std %s' % (rescale_proba, rescale_std))
        print('rescale unif proba %s, intens %s' % (rescale_unif_proba, rescale_unif_intens))
        print('noise proba %s, std %s' % (noise_proba, noise_std))
        print('indep noise proba %s, std %s' % (indep_noise_proba, indep_noise_std))
        print('elastic proba %s, alpha %s, sigma %s' % (elastic_proba, elastic_alpha, elastic_sigma))
        print('random waves proba %s, params %s' % (random_waves_proba, random_waves_params))
        print('random anti waves proba %s, params %s' % (random_anti_waves_proba, random_anti_waves_params))
        print("false spindles single cont proba %s, params %s" % (
            false_spindles_single_cont_proba, false_spindles_single_cont_params))

        if rescale_proba > 0:
            print('Applying gaussian rescaling augmentation')
            feat = augmentations.rescale_normal(
                feat, rescale_proba, rescale_std)

        if noise_proba > 0:
            print('Applying gaussian noise augmentation')
            feat = augmentations.gaussian_noise(
                feat, noise_proba, noise_std)

        if rescale_unif_proba > 0:
            print('Applying uniform rescaling augmentation')
            feat = augmentations.rescale_uniform(
                feat, rescale_unif_proba, rescale_unif_intens)

        if elastic_proba > 0:
            print('Applying elastic deformations')
            feat, label = augmentations.elastic_1d_deformation_wrapper(
                feat, label, elastic_proba, self.params[pkeys.FS],
                elastic_alpha, elastic_sigma)

        if indep_noise_proba > 0:
            print('Applying INDEPENDENT gaussian noise augmentation')
            feat = augmentations.independent_gaussian_noise(
                feat, indep_noise_proba, indep_noise_std)

        if random_anti_waves_proba > 0:
            print("Applying random anti waves augmentation")
            feat = augmentations.random_anti_waves_wrapper(
                feat, label, random_anti_waves_proba, self.params[pkeys.FS], random_anti_waves_params
            )

        if random_waves_proba > 0:
            print("Applying random waves augmentation")
            feat = augmentations.random_waves_wrapper(
                feat, label, random_waves_proba, self.params[pkeys.FS], random_waves_params)

        if false_spindles_single_cont_proba > 0:
            print("Applying false spindle single cont. augmentation")
            feat = augmentations.false_spindles_single_contamination_wrapper(
                feat, label, random_waves_proba, self.params[pkeys.FS], false_spindles_single_cont_params)

        return feat, label, mask

    def _model_fn(self):
        model_version = self.params[pkeys.MODEL_VERSION]
        checks.check_valid_value(
            model_version, 'model_version',
            [
                constants.DUMMY,
                constants.V1,
                constants.V4,
                constants.V5,
                constants.V6,
                constants.V7,
                constants.V8,
                constants.V9,
                constants.DEBUG,
                constants.V7lite,
                constants.V7litebig,
                constants.V10,
                constants.V11,
                constants.V12,
                constants.V13,
                constants.V14,
                constants.V15,
                constants.V16,
                constants.V17,
                constants.V18,
                constants.V19,
                constants.V20_INDEP,
                constants.V20_CONCAT,
                constants.V21,
                constants.V22,
                constants.V23,
                constants.V24,
                constants.V25,
                constants.V11_SKIP,
                constants.V19_SKIP,
                constants.V19_SKIP2,
                constants.V19_SKIP3,
                constants.V26,
                constants.V27,
                constants.V28,
                constants.V29,
                constants.V30,
                constants.V115,
                constants.V195,
                constants.V11G,
                constants.V19G,
                constants.V31,
                constants.V32,
                constants.V19P,
                constants.V33,
                constants.V34,
                constants.ATT01,
                constants.ATT02,
                constants.ATT03,
                constants.ATT04,
                constants.ATT04C,
                constants.V35,
                constants.V11_ABLATION,
                constants.V11_ABLATION_SCALED,
                constants.V11_D6K5,
                constants.V11_D8K3,
                constants.V11_D8K5,
                constants.V11_OUTRES,
                constants.V11_OUTPLUS,
                constants.V11_SHIELD,
                constants.V11_LITE,
                constants.V11_NORM,
                constants.V11_PR_1,
                constants.V11_PR_2P,
                constants.V11_PR_2C,
                constants.V11_PR_3P,
                constants.V11_PR_3C,
                constants.V11_LLC_STFT,
                constants.V11_LLC_STFT_1,
                constants.V11_LLC_STFT_2,
                constants.V11_LLC_STFT_3,
                constants.V19_LLC_STFT_2,
                constants.V19_LLC_STFT_3,
                constants.TCN01,
                constants.TCN02,
                constants.TCN03,
                constants.TCN04,
                constants.V19_FROZEN,
                constants.ATT05,
                constants.V19_VAR,
                constants.V19_NOISY,
                constants.A7_V1,
                constants.A7_V2,
                constants.A7_V3,
                constants.V11_BP,
                constants.V19_BP,
                constants.V11_LN,
                constants.V11_LN2,
                constants.V11_LN3,
                constants.V19_LN2,
                constants.V11_MK,
                constants.V11_MKD,
                constants.V11_MKD2,
                constants.V11_MKD2_STATMOD,
                constants.V11_MKD2_STATDOT,
                constants.V36,
                constants.V11_ATT,
                constants.V11_MKD2_EXPERTMOD,
                constants.V11_MKD2_EXPERTREG,
                constants.V11_MKD2_SWISH,
                constants.V41,
                constants.V42,
                constants.V43
             ])
        if model_version == constants.V1:
            model_fn = networks.wavelet_blstm_net_v1
        elif model_version == constants.V4:
            model_fn = networks.wavelet_blstm_net_v4
        elif model_version == constants.V5:
            model_fn = networks.wavelet_blstm_net_v5
        elif model_version == constants.V6:
            model_fn = networks.wavelet_blstm_net_v6
        elif model_version == constants.V7:
            model_fn = networks.wavelet_blstm_net_v7
        elif model_version == constants.V8:
            model_fn = networks.wavelet_blstm_net_v8
        elif model_version == constants.V9:
            model_fn = networks.wavelet_blstm_net_v9
        elif model_version == constants.V7lite:
            model_fn = networks.wavelet_blstm_net_v7_lite
        elif model_version == constants.V7litebig:
            model_fn = networks.wavelet_blstm_net_v7_litebig
        elif model_version == constants.V10:
            model_fn = networks.wavelet_blstm_net_v10
        elif model_version == constants.V11:
            model_fn = networks.wavelet_blstm_net_v11
        elif model_version == constants.V12:
            model_fn = networks.wavelet_blstm_net_v12
        elif model_version == constants.V13:
            model_fn = networks.wavelet_blstm_net_v13
        elif model_version == constants.V14:
            model_fn = networks.wavelet_blstm_net_v14
        elif model_version == constants.V15:
            model_fn = networks.wavelet_blstm_net_v15
        elif model_version == constants.V16:
            model_fn = networks.wavelet_blstm_net_v16
        elif model_version == constants.V17:
            model_fn = networks.wavelet_blstm_net_v17
        elif model_version == constants.V18:
            model_fn = networks.wavelet_blstm_net_v18
        elif model_version == constants.V19:
            model_fn = networks.wavelet_blstm_net_v19
        elif model_version == constants.V20_INDEP:
            model_fn = networks.wavelet_blstm_net_v20_indep
        elif model_version == constants.V20_CONCAT:
            model_fn = networks.wavelet_blstm_net_v20_concat
        elif model_version == constants.V21:
            model_fn = networks.wavelet_blstm_net_v21
        elif model_version == constants.V22:
            model_fn = networks.wavelet_blstm_net_v22
        elif model_version == constants.V23:
            model_fn = networks.wavelet_blstm_net_v23
        elif model_version == constants.V24:
            model_fn = networks.wavelet_blstm_net_v24
        elif model_version == constants.V25:
            model_fn = networks.wavelet_blstm_net_v25
        elif model_version == constants.V11_SKIP:
            model_fn = networks.wavelet_blstm_net_v11_skip
        elif model_version == constants.V19_SKIP:
            model_fn = networks.wavelet_blstm_net_v19_skip
        elif model_version == constants.V19_SKIP2:
            model_fn = networks.wavelet_blstm_net_v19_skip2
        elif model_version == constants.V19_SKIP3:
            model_fn = networks.wavelet_blstm_net_v19_skip3
        elif model_version == constants.V26:
            model_fn = networks.wavelet_blstm_net_v26
        elif model_version == constants.V27:
            model_fn = networks.wavelet_blstm_net_v27
        elif model_version == constants.V28:
            model_fn = networks.wavelet_blstm_net_v28
        elif model_version == constants.V29:
            model_fn = networks.wavelet_blstm_net_v29
        elif model_version == constants.V30:
            model_fn = networks.wavelet_blstm_net_30
        elif model_version == constants.V115:
            model_fn = networks.wavelet_blstm_net_v115
        elif model_version == constants.V195:
            model_fn = networks.wavelet_blstm_net_v195
        elif model_version == constants.V11G:
            model_fn = networks.wavelet_blstm_net_v11g
        elif model_version == constants.V19G:
            model_fn = networks.wavelet_blstm_net_v19g
        elif model_version == constants.V31:
            model_fn = networks.wavelet_blstm_net_v31
        elif model_version == constants.V32:
            model_fn = networks.wavelet_blstm_net_v32
        elif model_version == constants.V19P:
            model_fn = networks.wavelet_blstm_net_v19p
        elif model_version == constants.V33:
            model_fn = networks.wavelet_blstm_net_v33
        elif model_version == constants.V34:
            model_fn = networks.wavelet_blstm_net_v34
        elif model_version == constants.ATT01:
            model_fn = networks.wavelet_blstm_net_att01
        elif model_version == constants.ATT02:
            model_fn = networks.wavelet_blstm_net_att02
        elif model_version == constants.ATT03:
            model_fn = networks.wavelet_blstm_net_att03
        elif model_version == constants.ATT04:
            model_fn = networks.wavelet_blstm_net_att04
        elif model_version == constants.ATT04C:
            model_fn = networks.wavelet_blstm_net_att04c
        elif model_version == constants.V35:
            model_fn = networks.wavelet_blstm_net_v35
        elif model_version == constants.V11_ABLATION:
            model_fn = networks.wavelet_blstm_net_v11_ablation
        elif model_version == constants.V11_ABLATION_SCALED:
            model_fn = networks.wavelet_blstm_net_v11_ablation_scaled
        elif model_version == constants.V11_D6K5:
            model_fn = networks.wavelet_blstm_net_v11_d6k5
        elif model_version == constants.V11_D8K3:
            model_fn = networks.wavelet_blstm_net_v11_d8k3
        elif model_version == constants.V11_D8K5:
            model_fn = networks.wavelet_blstm_net_v11_d8k5
        elif model_version == constants.V11_OUTRES:
            model_fn = networks.wavelet_blstm_net_v11_outres
        elif model_version == constants.V11_OUTPLUS:
            model_fn = networks.wavelet_blstm_net_v11_outplus
        elif model_version == constants.V11_SHIELD:
            model_fn = networks.wavelet_blstm_net_v11_shield
        elif model_version == constants.V11_LITE:
            model_fn = networks.wavelet_blstm_net_v11_lite
        elif model_version == constants.V11_NORM:
            model_fn = networks.wavelet_blstm_net_v11_norm
        elif model_version == constants.V11_PR_1:
            model_fn = networks.wavelet_blstm_net_v11_pr_1
        elif model_version == constants.V11_PR_2P:
            model_fn = networks.wavelet_blstm_net_v11_pr_2p
        elif model_version == constants.V11_PR_2C:
            model_fn = networks.wavelet_blstm_net_v11_pr_2c
        elif model_version == constants.V11_PR_3P:
            model_fn = networks.wavelet_blstm_net_v11_pr_3p
        elif model_version == constants.V11_PR_3C:
            model_fn = networks.wavelet_blstm_net_v11_pr_3c
        elif model_version == constants.V11_LLC_STFT:
            model_fn = networks.wavelet_blstm_net_v11_llc_stft
        elif model_version == constants.V11_LLC_STFT_1:
            model_fn = networks.wavelet_blstm_net_v11_llc_stft_1
        elif model_version == constants.V11_LLC_STFT_2:
            model_fn = networks.wavelet_blstm_net_v11_llc_stft_2
        elif model_version == constants.V11_LLC_STFT_3:
            model_fn = networks.wavelet_blstm_net_v11_llc_stft_3
        elif model_version == constants.V19_LLC_STFT_2:
            model_fn = networks.wavelet_blstm_net_v19_llc_stft_2
        elif model_version == constants.V19_LLC_STFT_3:
            model_fn = networks.wavelet_blstm_net_v19_llc_stft_3
        elif model_version == constants.TCN01:
            model_fn = networks.wavelet_blstm_net_tcn01
        elif model_version == constants.TCN02:
            model_fn = networks.wavelet_blstm_net_tcn02
        elif model_version == constants.TCN03:
            model_fn = networks.wavelet_blstm_net_tcn03
        elif model_version == constants.TCN04:
            model_fn = networks.wavelet_blstm_net_tcn04
        elif model_version == constants.V19_FROZEN:
            model_fn = networks.wavelet_blstm_net_v19_frozen
        elif model_version == constants.ATT05:
            model_fn = networks_v2.wavelet_blstm_net_att05
        elif model_version == constants.V19_VAR:
            model_fn = networks.wavelet_blstm_net_v19_var
        elif model_version == constants.V19_NOISY:
            model_fn = networks.wavelet_blstm_net_v19_noisy
        elif model_version == constants.A7_V1:
            model_fn = networks_v2.deep_a7_v1
        elif model_version == constants.A7_V2:
            model_fn = networks_v2.deep_a7_v2
        elif model_version == constants.A7_V3:
            model_fn = networks_v2.deep_a7_v3
        elif model_version == constants.V11_BP:
            model_fn = networks_v2.wavelet_blstm_net_v11_bp
        elif model_version == constants.V19_BP:
            model_fn = networks_v2.wavelet_blstm_net_v19_bp
        elif model_version == constants.V11_LN:
            model_fn = networks_v2.wavelet_blstm_net_v11_ln
        elif model_version == constants.V11_LN2:
            model_fn = networks_v2.wavelet_blstm_net_v11_ln2
        elif model_version == constants.V11_LN3:
            model_fn = networks_v2.wavelet_blstm_net_v11_ln3
        elif model_version == constants.V19_LN2:
            model_fn = networks_v2.wavelet_blstm_net_v19_ln2
        elif model_version == constants.V11_MK:
            model_fn = networks_v2.wavelet_blstm_net_v11_mk
        elif model_version == constants.V11_MKD:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd
        elif model_version == constants.V11_MKD2:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd2
        elif model_version == constants.V11_MKD2_STATMOD:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd2_statmod
        elif model_version == constants.V11_MKD2_STATDOT:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd2_statdot
        elif model_version == constants.V36:
            model_fn = networks_v2.wavelet_blstm_net_v36
        elif model_version == constants.V11_ATT:
            model_fn = networks_v2.wavelet_blstm_net_v11_att
        elif model_version == constants.V11_MKD2_EXPERTMOD:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd2_expertmod
        elif model_version == constants.V11_MKD2_EXPERTREG:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd2_expertreg
        elif model_version == constants.V11_MKD2_SWISH:
            model_fn = networks_v2.wavelet_blstm_net_v11_mkd2_swish
        elif model_version == constants.V41:
            model_fn = networks_v2.wavelet_blstm_net_v41
        elif model_version == constants.V42:
            model_fn = networks_v2.wavelet_blstm_net_v42
        elif model_version == constants.V43:
            model_fn = networks_v2.wavelet_blstm_net_v43
        elif model_version == constants.DEBUG:
            model_fn = networks.debug_net
        else:
            model_fn = networks.dummy_net

        logits, probabilities, other_outputs_dict = model_fn(self.feats, self.params, self.training_ph)
        return logits, probabilities, other_outputs_dict

    def _loss_fn(self):
        type_loss = self.params[pkeys.TYPE_LOSS]
        checks.check_valid_value(
            type_loss, 'type_loss',
            [
                constants.CROSS_ENTROPY_LOSS,
                constants.DICE_LOSS,
                constants.FOCAL_LOSS,
                constants.WORST_MINING_LOSS,
                constants.WORST_MINING_V2_LOSS,
                constants.CROSS_ENTROPY_NEG_ENTROPY_LOSS,
                constants.CROSS_ENTROPY_SMOOTHING_LOSS,
                constants.CROSS_ENTROPY_HARD_CLIP_LOSS,
                constants.CROSS_ENTROPY_SMOOTHING_CLIP_LOSS,
                constants.MOD_FOCAL_LOSS,
                constants.CROSS_ENTROPY_BORDERS_LOSS,
                constants.CROSS_ENTROPY_BORDERS_IND_LOSS,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS_HARD,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS_SOFT,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS_V2,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS_V3,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS_V4,
                constants.HINGE_LOSS,
                constants.WEIGHTED_CROSS_ENTROPY_LOSS_V5,
                constants.CROSS_ENTROPY_LOSS_WITH_LOGITS_REG,
                constants.CROSS_ENTROPY_LOSS_WITH_SELF_SUPERVISION,
                constants.MASKED_SOFT_FOCAL_LOSS
            ])

        if type_loss == constants.CROSS_ENTROPY_LOSS:
            loss, loss_summ = losses.cross_entropy_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS])
        elif type_loss == constants.FOCAL_LOSS:
            loss, loss_summ = losses.focal_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.FOCUSING_PARAMETER])
        elif type_loss == constants.WORST_MINING_LOSS:
            loss, loss_summ = losses.worst_mining_loss_fn(
                self.logits, self.labels,
                self.params[pkeys.WORST_MINING_FACTOR_NEGATIVE],
                self.params[pkeys.WORST_MINING_MIN_NEGATIVE])
        elif type_loss == constants.WORST_MINING_V2_LOSS:
            loss, loss_summ = losses.worst_mining_v2_loss_fn(
                self.logits, self.labels,
                self.params[pkeys.WORST_MINING_FACTOR_NEGATIVE],
                self.params[pkeys.WORST_MINING_MIN_NEGATIVE])
        elif type_loss == constants.CROSS_ENTROPY_NEG_ENTROPY_LOSS:
            loss, loss_summ = losses.cross_entropy_negentropy_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.NEG_ENTROPY_PARAMETER])
        elif type_loss == constants.CROSS_ENTROPY_SMOOTHING_LOSS:
            loss, loss_summ = losses.cross_entropy_smoothing_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.SOFT_LABEL_PARAMETER])
        elif type_loss == constants.CROSS_ENTROPY_SMOOTHING_CLIP_LOSS:
            loss, loss_summ = losses.cross_entropy_smoothing_clip_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.SOFT_LABEL_PARAMETER])
        elif type_loss == constants.CROSS_ENTROPY_HARD_CLIP_LOSS:
            loss, loss_summ = losses.cross_entropy_hard_clip_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.SOFT_LABEL_PARAMETER])
        elif type_loss == constants.MOD_FOCAL_LOSS:
            loss, loss_summ = losses.mod_focal_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.FOCUSING_PARAMETER],
                self.params[pkeys.MIS_WEIGHT_PARAMETER])
        elif type_loss == constants.CROSS_ENTROPY_BORDERS_LOSS:
            loss, loss_summ = losses.cross_entropy_loss_borders_fn(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH])
        elif type_loss == constants.CROSS_ENTROPY_BORDERS_IND_LOSS:
            loss, loss_summ = losses.cross_entropy_loss_borders_ind_fn(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS:
            loss, loss_summ = losses.weighted_cross_entropy_loss(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH],
                self.params[pkeys.FOCUSING_PARAMETER],
                self.params[pkeys.MIS_WEIGHT_PARAMETER],
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.MIX_WEIGHTS_STRATEGY])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS_HARD:
            loss, loss_summ = losses.weighted_cross_entropy_loss_hard(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH],
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.MIX_WEIGHTS_STRATEGY])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS_SOFT:
            loss, loss_summ = losses.weighted_cross_entropy_loss_soft(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH],
                self.params[pkeys.SOFT_LABEL_PARAMETER],
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.MIX_WEIGHTS_STRATEGY])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS_V2:
            loss, loss_summ = losses.weighted_cross_entropy_loss_v2(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH],
                self.params[pkeys.FOCUSING_PARAMETER],
                self.params[pkeys.MIS_WEIGHT_PARAMETER],
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.MIX_WEIGHTS_STRATEGY],
                self.params[pkeys.PREDICTION_VARIABILITY_REGULARIZER])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS_V3:
            loss, loss_summ = losses.weighted_cross_entropy_loss_v3(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH],
                self.params[pkeys.FOCUSING_PARAMETER],
                self.params[pkeys.MIS_WEIGHT_PARAMETER],
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.MIX_WEIGHTS_STRATEGY],
                self.params[pkeys.PREDICTION_VARIABILITY_REGULARIZER],
                self.params[pkeys.PREDICTION_VARIABILITY_LAG])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS_V4:
            loss, loss_summ = losses.weighted_cross_entropy_loss_v4(
                self.logits, self.labels,
                self.params[pkeys.BORDER_WEIGHT_AMPLITUDE],
                self.params[pkeys.BORDER_WEIGHT_HALF_WIDTH],
                self.params[pkeys.FOCUSING_PARAMETER],
                self.params[pkeys.MIS_WEIGHT_PARAMETER],
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.MIX_WEIGHTS_STRATEGY],
                self.params[pkeys.PREDICTION_VARIABILITY_REGULARIZER],
                self.params[pkeys.PREDICTION_VARIABILITY_LAG])
        elif type_loss == constants.WEIGHTED_CROSS_ENTROPY_LOSS_V5:
            loss, loss_summ = losses.weighted_cross_entropy_loss_v5(
                self.logits, self.labels,
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.SOFT_FOCAL_GAMMA], self.params[pkeys.SOFT_FOCAL_EPSILON],
                self.params[pkeys.ANTIBORDER_AMPLITUDE], self.params[pkeys.ANTIBORDER_HALF_WIDTH]
            )
        elif type_loss == constants.CROSS_ENTROPY_LOSS_WITH_LOGITS_REG:
            loss, loss_summ = losses.cross_entropy_loss_with_logits_reg_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.LOGITS_REG_TYPE], self.params[pkeys.LOGITS_REG_WEIGHT]
            )
        elif type_loss == constants.HINGE_LOSS:
            loss, loss_summ = losses.hinge_loss_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS])
        elif type_loss == constants.CROSS_ENTROPY_LOSS_WITH_SELF_SUPERVISION:
            loss, loss_summ = losses.cross_entropy_loss_with_self_supervision_fn(
                self.logits, self.labels, self.params[pkeys.CLASS_WEIGHTS],
                self.other_outputs_dict["regression_loss"],
                self.params[pkeys.EXPERT_BRANCH_REGRESSION_LOSS_COEFFICIENT])
        elif type_loss == constants.MASKED_SOFT_FOCAL_LOSS:
            loss, loss_summ = losses.masked_soft_focal_loss(
                self.logits, self.labels, self.masks,
                self.params[pkeys.CLASS_WEIGHTS],
                self.params[pkeys.SOFT_FOCAL_GAMMA], self.params[pkeys.SOFT_FOCAL_EPSILON])
        else:
            loss, loss_summ = losses.dice_loss_fn(
                self.probabilities[..., 1], self.labels)
        return loss, loss_summ

    def _optimizer_fn(self):
        type_optimizer = self.params[pkeys.TYPE_OPTIMIZER]
        checks.check_valid_value(
            type_optimizer, 'type_optimizer',
            [
                constants.ADAM_OPTIMIZER,
                constants.SGD_OPTIMIZER,
                constants.RMSPROP_OPTIMIZER,
                constants.ADAM_W_OPTIMIZER
            ])

        if type_optimizer == constants.ADAM_OPTIMIZER:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.adam_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[pkeys.CLIP_NORM])
        elif type_optimizer == constants.ADAM_W_OPTIMIZER:
            train_step, reset_optimizer_op, grad_norm_summ = optimizers.adam_w_optimizer_fn(
                self.loss, self.learning_rate, self.weight_decay,
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
            tp, fp, fn = metrics.confusion_matrix(self.logits, self.labels, self.masks)
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

    def _split_train(self, x_train, y_train, m_train):
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
            m_train_1 = m_train[zero_activity_idx]
            # Pages with activity
            x_train_2 = x_train[exists_activity_idx]
            y_train_2 = y_train[exists_activity_idx]
            m_train_2 = m_train[exists_activity_idx]
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
            m_train_1 = m_train[low_activity_idx]
            # Pages with high activity
            x_train_2 = x_train[high_activity_idx]
            y_train_2 = y_train[high_activity_idx]
            m_train_2 = m_train[high_activity_idx]
            print('Pages with low activity:', x_train_1.shape)
            print('Pages with high activity:', x_train_2.shape)

        return x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2

