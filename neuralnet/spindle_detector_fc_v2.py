from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import datetime

from cwt_layer import complex_morlet_layer
from context_net import context_net

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


class SpindleDetectorFC(object):

    def __init__(self, params):
        # Training parameters
        self.dropout_rate = 0.3
        self.pos_weight = 3
        self.learning_rate = 0.001
        self.batch_size = 64

        # General parameters
        self.context = params['context']
        self.factor_border = params['factor_border']
        self.mark_smooth = params['mark_smooth']
        self.fs = params["fs"]

        # CWT parameters
        self.cwt_stride = 2
        self.fc_array = np.array([1, 1, 1, 1])
        self.fb_array = np.array([0.5, 1, 1.5, 2])
        self.n_scales = 32
        self.upper_freq = 40
        self.lower_freq = 2

        # Some static values
        self.segment_size = int((self.factor_border + 1) * self.context * self.fs)
        self.context_size = int(self.context * self.fs / self.cwt_stride)
        self.context_start = int(self.factor_border * self.context * self.fs / (2*self.cwt_stride))
        self.context_end = self.context_start + self.context_size
        self.border_size = self.context_start*self.cwt_stride

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_name = 'fc_v1'
        self.model_path = 'results/' + self.model_name + '_' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summaries/'

        # Model initialization
        with tf.name_scope("input"):
            self.features_ph = tf.placeholder(shape=[None, 1, self.segment_size, 1],
                                              dtype=tf.float32, name="features")
            self.labels_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="labels")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.logits, self.prediction = self._model_init()
        self.loss = self._loss_init()
        self.train_step = self._optimizer_init()
        self.metrics, self.metrics_upd, self.metrics_init = self._batch_metrics_init()

        # Session
        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.train_writer = tf.summary.FileWriter(self.tb_path + 'train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.tb_path + 'val')

        # Initialization of variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, dataset, max_it, stat_every=50, save_every=5000):
        merged = tf.summary.merge_all()

        start_time = time.time()
        print("Beginning training of " + self.model_name)
        print("Beginning training of " + self.model_name, file=open(self.model_path + 'train.log', 'w'))

        # features, labels = dataset.next_batch(batch_size=64,
        #                                       segment_size=self.segment_size,
        #                                       mark_smooth=self.mark_smooth,
        #                                       dataset="TRAIN")
        # cwt_image = self.sess.run(self.inputs_cwt,
        #                           feed_dict={self.features_ph: features,
        #                                      self.labels_ph: labels})
        # # show first 2 on the batch, only fb[0]
        # for i in range(2):
        #     plt.figure(figsize=(15, 3))
        #     plt.imshow(cwt_image[i, :, :, 0], interpolation='none', cmap=cm.inferno, aspect='auto')
        #     plt.title("TF")
        #     plt.show()

        for it in range(1, max_it + 1):
            features, labels = dataset.next_batch(batch_size=self.batch_size,
                                                  segment_size=self.segment_size,
                                                  mark_smooth=self.mark_smooth,
                                                  sub_set="TRAIN")
            feed_dict = {self.features_ph: features, self.labels_ph: labels, self.is_training: True}
            self.sess.run(self.train_step, feed_dict=feed_dict)

            if it % stat_every == 0:
                # Training stat
                feed_dict = {self.features_ph: features, self.labels_ph: labels, self.is_training: False}
                self.sess.run(self.metrics_init)
                train_loss, _ = self.sess.run([self.loss, self.metrics_upd], feed_dict=feed_dict)
                summ, metrics = self.sess.run([merged, self.metrics], feed_dict=feed_dict)
                train_pre = metrics["precision"]
                train_rec = metrics["recall"]
                self.train_writer.add_summary(summ, it)
                # cwt_image = self.sess.run(self.inputs_cwt,
                #                           feed_dict={self.features_ph: features,
                #                                      self.labels_ph: labels})
                # Validation stat
                features, labels = dataset.next_batch(batch_size=self.batch_size,
                                                      segment_size=self.segment_size,
                                                      mark_smooth=self.mark_smooth,
                                                      sub_set="VAL")
                feed_dict = {self.features_ph: features, self.labels_ph: labels, self.is_training: False}
                self.sess.run(self.metrics_init)
                val_loss, _ = self.sess.run([self.loss, self.metrics_upd], feed_dict=feed_dict)
                summ, metrics = self.sess.run([merged, self.metrics], feed_dict=feed_dict)
                val_pre = metrics["precision"]
                val_rec = metrics["recall"]
                self.val_writer.add_summary(summ, it)

                time_usage = time.time() - start_time
                print("step %i/%i: train loss %f pre %f rec %f - val loss %f pre %f rec %f - time %f s"
                      % (it, max_it, train_loss, train_pre, train_rec, val_loss, val_pre, val_rec, time_usage),
                      flush=True)
                print("step %i/%i: train loss %f pre %f rec %f - val loss %f pre %f rec %f - time %f s"
                      % (it, max_it, train_loss, train_pre, train_rec, val_loss, val_pre, val_rec, time_usage),
                      flush=True, file=open(self.model_path + 'train.log', 'a'))

                # show first 2 on the batch, only fb[0]
                # for i in range(2):
                #     plt.figure(figsize=(15, 3))
                #     plt.imshow(cwt_image[i, :, :, 0], interpolation='none', cmap=cm.inferno, aspect='auto')
                #     plt.title("TF")
                #     plt.show()

            if it % save_every == 0:
                save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=max_it)
        print("Model saved to: %s" % save_path)
        print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        # Training set overall evaluation bysample
        metrics = self.subset_evaluation(dataset, "TRAIN")
        train_pre = metrics["precision"]
        train_rec = metrics["recall"]
        # Validation set overall evaluation bysample
        metrics = self.subset_evaluation(dataset, "VAL")
        val_pre = metrics["precision"]
        val_rec = metrics["recall"]
        # Testing set overall evaluation bysample
        metrics = self.subset_evaluation(dataset, "TEST")
        test_pre = metrics["precision"]
        test_rec = metrics["recall"]
        # Show results
        print("TRAIN final by-sample: precision %f - recall %f" % (train_pre, train_rec))
        print("TRAIN final by-sample: precision %f - recall %f" % (train_pre, train_rec),
              file=open(self.model_path + 'train.log', 'a'))
        print("VAL final by-sample: precision %f - recall %f" % (val_pre, val_rec))
        print("VAL final by-sample: precision %f - recall %f" % (val_pre, val_rec),
              file=open(self.model_path + 'train.log', 'a'))
        print("TEST final by-sample: precision %f - recall %f" % (test_pre, test_rec))
        print("TEST final by-sample: precision %f - recall %f" % (test_pre, test_rec),
              file=open(self.model_path + 'train.log', 'a'))
        # Closing savers
        self.train_writer.close()
        self.val_writer.close()

        # Total time
        time_usage = time.time() - start_time
        print("Time usage: " + str(time_usage) + " [s]")
        print("Time usage: " + str(time_usage) + " [s]", file=open(self.model_path + 'train.log', 'a'))

    def _model_init(self):
        with tf.name_scope("model"):
            reuse = False
            inputs = self.features_ph
            # Convert signal into CWT
            inputs_cwt = complex_morlet_layer(inputs, self.fc_array, self.fb_array, self.fs,
                                              self.lower_freq, self.upper_freq, self.n_scales,
                                              border_crop=self.border_size, stride=self.cwt_stride)

            # Normalize CWT
            inputs_cwt_bn = tf.layers.batch_normalization(inputs=inputs_cwt, training=self.is_training,
                                                          reuse=reuse, name="bn_1")

            s_t = self._slicing_init(inputs_cwt_bn)  # Central slice
            # c_t = context_net(inputs_cwt_bn, training=self.is_training)  # Context vector

            # concat_inputs = tf.concat([s_t, c_t], 1)
            # h_t = self._fc_net_init(concat_inputs)  # Fully Connected baseline
            h_t = self._fc_net_init(s_t)

            # Final Classification
            with tf.variable_scope("output"):
                logits = tf.layers.dense(inputs=h_t, units=1, activation=None, name="logits", reuse=reuse)
                prediction = tf.sigmoid(logits, name="prediction")

        return logits, prediction

    def _slicing_init(self, inputs, reuse=False):
        with tf.variable_scope("slicing"):
            central_pos = int(self.context_size/2)
            central_slice = inputs[:, :, central_pos:(central_pos+1), :]
            s_t = tf.layers.conv2d(inputs=central_slice, filters=1, kernel_size=1, activation=tf.nn.relu,
                                   padding="same", name="c_1x1", reuse=reuse)
            s_t_flat = tf.squeeze(s_t, axis=[2, 3])
        return s_t_flat

    def _fc_net_init(self, inputs, reuse=False):
        with tf.variable_scope("fc_net"):
            inputs_drop = tf.layers.dropout(inputs=inputs, rate=self.dropout_rate,
                                            training=self.is_training, name="drop_1")
            # inputs_bn = tf.layers.batch_normalization(inputs=inputs, training=self.is_training,
            #                                           name="bn_1", reuse=reuse)
            fc_1 = tf.layers.dense(inputs=inputs_drop, units=256, activation=tf.nn.relu, name="fc_1", reuse=reuse)
            fc_1_drop = tf.layers.dropout(inputs=fc_1, rate=self.dropout_rate,
                                          training=self.is_training, name="drop_2")
            # fc_1_bn = tf.layers.batch_normalization(inputs=fc_1, training=self.is_training,
            #                                         name="bn_2", reuse=reuse)
            fc_2 = tf.layers.dense(inputs=fc_1_drop, units=128, activation=tf.nn.relu, name="fc_2", reuse=reuse)
        return fc_2

    def _loss_init(self):
        # Suggested: L2 regularization and weights for balancing
        with tf.name_scope("loss"):
            labels = tf.expand_dims(self.labels_ph, 1)  # Make it 2D tensor
            # byexample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
            #                                                          labels=labels)
            byexample_loss = tf.nn.weighted_cross_entropy_with_logits(
                targets=labels,
                logits=self.logits,
                pos_weight=self.pos_weight,
            )
            loss = tf.reduce_mean(byexample_loss)
            tf.summary.scalar('loss', loss)
        return loss

    def _optimizer_init(self):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For BN
            with tf.control_dependencies(update_ops):
                train_step = optimizer.minimize(self.loss)
        return train_step

    def _batch_metrics_init(self):
        with tf.variable_scope("metrics"):
            labels = tf.expand_dims(self.labels_ph, 1)  # Make it 2D tensor
            prediction_bin = tf.greater(self.prediction, tf.constant(0.5))
            fp, fp_upd = tf.metrics.false_positives(labels=labels, predictions=prediction_bin, name="fp_metric")
            fn, fn_upd = tf.metrics.false_negatives(labels=labels, predictions=prediction_bin, name="fn_metric")
            tp, tp_upd = tf.metrics.true_positives(labels=labels, predictions=prediction_bin, name="tp_metric")
            precision = tp / (tp + fp)
            tf.summary.scalar('batch_precision', precision)
            recall = tp / (tp + fn)
            tf.summary.scalar('batch_recall', recall)
            f1_score = 2 * precision * recall / (precision + recall)
            tf.summary.scalar('batch_f1_score', f1_score)
            metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
            metrics_upd = tf.group(fp_upd, fn_upd, tp_upd)

            fp_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics/fp_metric")
            fn_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics/fn_metric")
            tp_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics/tp_metric")
            metrics_init = tf.variables_initializer(var_list=fp_vars+fn_vars+tp_vars)
        return metrics, metrics_upd, metrics_init

    def subset_evaluation(self, dataset, sub_set):
        start_time = time.time()
        print("Evaluating " + sub_set)
        batch_size = 200
        tau = 10
        batch_eff = tau*batch_size
        data_list = dataset.get_sub_set(sub_set)
        epoch_size = dataset.get_epoch_size()
        n_it_per_epoch = int(epoch_size/batch_eff)
        self.sess.run(self.metrics_init)
        ind_dict = data_list[0]
        #for ind_dict in data_list:
        for epoch in ind_dict["epochs"]:
            offset = epoch * epoch_size
            for counter in range(n_it_per_epoch):
                central_samples = np.arange(offset + counter*batch_eff, offset + (counter+1)*batch_eff, tau)
                # Build batch
                features = np.zeros((batch_size, 1, self.segment_size, 1), dtype=np.float32)
                labels = np.zeros(batch_size, dtype=np.float32)
                for i in range(len(central_samples)):
                    sample = central_samples[i]
                    # Get signal segment
                    sample_start = sample - int(self.segment_size / 2)
                    sample_end = sample + int(self.segment_size / 2)
                    features[i, 0, :, 0] = ind_dict['signal'][sample_start:sample_end]
                    # Get mark, with an optional smoothing
                    smooth_start = sample - int(np.floor(self.mark_smooth / 2))
                    smooth_end = smooth_start + self.mark_smooth
                    mark_array = ind_dict['marks'][smooth_start:smooth_end]
                    smooth_mark = np.mean(mark_array)
                    labels[i] = smooth_mark
                # Evaluate batch
                feed_dict = {self.features_ph: features, self.labels_ph: labels, self.is_training: False}
                self.sess.run([self.metrics_upd], feed_dict=feed_dict)
        print("Register processed. Time Elapsed: " + str(time.time()-start_time) + " [s]")
        metrics = self.sess.run(self.metrics)
        return metrics


