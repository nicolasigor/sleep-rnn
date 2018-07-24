from __future__ import division
import tensorflow as tf
import numpy as np
import pywt
import time
import datetime

import dataload
import transform


class Detector(object):
    def __init__(self, params):

        # General parameters
        self.channel = params['channel']
        self.dur_epoch = params['dur_epoch']
        self.n2_val = params['n2_val']
        self.context = params['context']
        self.factor_border = params['factor_border']
        self.mark_smooth = params['mark_smooth']
        self.percentile = params['percentile']
        self.fs = params['fs']

        # CWT parameters
        self.fc = 1
        self.fb_array = np.array([0.5, 1, 1.5, 2])
        self.n_scales = 32
        self.upper_freq = 40
        self.lower_freq = 2

        # Some static values
        self.epoch_size = self.dur_epoch * self.fs
        self.segment_size = (self.factor_border + 1) * self.context * self.fs
        self.context_size = int(self.context * self.fs)
        self.context_start = int(self.factor_border * self.context * self.fs / 2)
        self.context_end = self.context_start + self.context_size

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_name = 'ss_detector_fc_1'
        self.model_path = 'results/' + self.model_name + ' ' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summaries/'

        # Model initialization
        with tf.name_scope("input"):
            self.features_ph = tf.placeholder(shape=[None, self.n_scales, self.context_size, self.fb_array.shape[0]],
                                              dtype=tf.float32, name="features")
            self.labels_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="labels")
        self.logits, self.prob = self._model_init()
        self.loss = self._loss_init()
        self.train_step = self._optimizer_init()

        # Session
        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.train_writer = tf.summary.FileWriter(self.tb_path + 'train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.tb_path + 'val')
        tf.summary.scalar('loss', self.loss)

        # Initialization of variables
        self.sess.run(tf.global_variables_initializer())

    def train(self, train_path_list, val_path_list, max_it, stat_every=50, save_every=200):

        merged = tf.summary.merge_all()

        # Load data
        print("Loading training set")
        data_train = self.load_data(train_path_list)
        print("Loading validation set")
        data_val = self.load_data(val_path_list)

        start_time = time.time()
        print("Beginning training of " + self.model_name)
        print("Beginning training of " + self.model_name, file=open(self.model_path + 'train.log', 'w'))
        it = 0

        # stat_every = np.max([int(max_it/100), 1])
        # save_every = np.max([int(max_it/5), 1])

        for it in range(1, max_it+1):
            features, labels = self.sample_minibatch(data_train, batch_size=64)
            features, _ = self.get_cwt_minibatch(features)
            self.sess.run(self.train_step,
                          feed_dict={self.features_ph: features,
                                     self.labels_ph: labels})
            if it % stat_every == 0:
                # Training stat
                train_loss, summ = self.sess.run([self.loss, merged],
                                                 feed_dict={self.features_ph: features,
                                                            self.labels_ph: labels})
                self.train_writer.add_summary(summ, it)
                # Validation stat
                features, labels = self.sample_minibatch(data_val, batch_size=64)
                features, _ = self.get_cwt_minibatch(features)
                val_loss, summ = self.sess.run([self.loss, merged],
                                               feed_dict={self.features_ph: features,
                                                          self.labels_ph: labels})
                self.val_writer.add_summary(summ, it)

                time_usage = time.time() - start_time
                print("Iteration %i/%i: train loss %f - val loss %f - time %f s"
                      % (it, max_it, train_loss, val_loss, time_usage), flush=True)
                print("Iteration %i/%i: train loss %f - val loss %f - time %f s"
                      % (it, max_it, train_loss, val_loss, time_usage),
                      flush=True, file=open(self.model_path + 'train.log', 'a'))

            if it % save_every == 0:
                save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
        print("Model saved to: %s" % save_path)
        print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        # Closing savers
        self.train_writer.close()
        self.val_writer.close()

        # Total time
        time_usage = time.time() - start_time
        print("Time usage: " + str(time_usage) + " [s]")
        print("Time usage: " + str(time_usage) + " [s]", file=open(self.model_path + 'train.log', 'a'))

    def evaluate(self, eval_path_list):
        data = self.load_data(eval_path_list)

    def predict(self, data_path_list):
        data = self.load_data(data_path_list, with_marks=False)

    def load_params(self):
        pass

    def load_data(self, data_path_list, with_marks=True):
        data_list = []
        n_data = len(data_path_list)
        start = time.time()
        for i in range(n_data):
            # Read EEG Signal
            path_edf_file = data_path_list[i]['file_edf']
            signal, _ = dataload.read_eeg(path_edf_file, self.channel)

            # Read states
            path_states_file = data_path_list[i]['file_states']
            states = dataload.read_states(path_states_file)

            # Find n2 epochs (individually)
            n2_epochs = np.where(states == self.n2_val)[0]
            # Drop first and last epoch of the whole registers if they where selected
            last_state = states.shape[0] - 1
            n2_epochs = n2_epochs[(n2_epochs != 0) & (n2_epochs != last_state)]
            # Clip-Normalize eeg signal
            epoch_size = self.dur_epoch * self.fs
            n2_list = []
            for epoch in n2_epochs:
                sample_start = epoch * epoch_size
                sample_end = (epoch + 1) * epoch_size
                n2_signal = signal[sample_start:sample_end]
                n2_list.append(n2_signal)
            n2_signal = np.concatenate(n2_list, axis=0)

            data = n2_signal
            thr = np.percentile(np.abs(data), self.percentile)
            data[np.abs(data) > thr] = float('nan')
            data_mean = np.nanmean(data)
            data_std = np.nanstd(data)

            signal = (np.clip(signal, -thr, thr) - data_mean) / data_std

            # Save data
            ind_dict = {'signal': np.array(signal),
                        'epochs': np.array(n2_epochs)}

            # Read Expert marks
            if with_marks:
                path_marks_file = data_path_list[i]['file_marks']
                marks = dataload.read_marks(path_marks_file, self.channel)
                marks = transform.inter2seq(marks, 0, len(signal) - 1)  # make 0/1
                ind_dict.update({'marks': np.array(marks)})

            data_list.append(ind_dict)
            print(str(i+1) + '/' + str(n_data) + ' ready, Time Elapsed: ' + str(time.time() - start) + ' [s]')

        print(len(data_list), ' records have been read.')
        return data_list

    def sample_minibatch(self, data_list, batch_size=32):
        features = np.zeros((batch_size, int(self.segment_size)))
        labels = np.zeros((batch_size, 1))

        n_data = len(data_list)
        ind_choice = np.random.choice(range(n_data), batch_size, replace=True)
        for i in range(batch_size):
            ind_dict = data_list[ind_choice[i]]
            # Choose a random epoch
            epoch = np.random.choice(ind_dict['epochs'])
            offset = epoch * self.epoch_size
            # Choose a random timestep in that epoch
            central_sample = np.random.choice(range(self.epoch_size))
            central_sample = offset + central_sample
            # Get signal segment
            sample_start = central_sample - int(self.segment_size / 2)
            sample_end = central_sample + int(self.segment_size / 2)
            features[i, :] = ind_dict['signal'][sample_start:sample_end]
            # Get mark, with an optional smoothing
            smooth_start = central_sample - int(np.floor(self.mark_smooth / 2))
            smooth_end = smooth_start + self.mark_smooth
            smooth_mark = np.mean(ind_dict['marks'][smooth_start:smooth_end])
            labels[i, :] = smooth_mark

        return features, labels

    def get_cwt_minibatch(self, signal_minibatch):
        # Generate initial and last scale
        s_0 = self.fs / self.upper_freq
        s_n = self.fs / self.lower_freq
        # Generate the array of scales
        base = np.power(s_n / s_0, 1 / (self.n_scales - 1))
        scales = s_0 * np.power(base, range(self.n_scales))
        # Set wavelet
        w = pywt.ContinuousWavelet('cmor')
        w.center_frequency = self.fc

        # Compute scalograms
        batch_size = signal_minibatch.shape[0]
        features_cwt = np.zeros((batch_size, self.n_scales, self.context_size, self.fb_array.shape[0]))
        for i in range(batch_size):
            for j in range(self.fb_array.shape[0]):
                w.bandwidth_frequency = self.fb_array[j]
                coef, freqs = pywt.cwt(signal_minibatch[i, :], scales, w, 1 / self.fs)
                abs_coef = np.abs(coef[:, self.context_start:self.context_end])
                # Spectrum flattening
                abs_coef = abs_coef * freqs[:, np.newaxis]
                features_cwt[i, :, :, j] = abs_coef

        return features_cwt, freqs

    def _model_init(self):
        with tf.name_scope("model"):
            inputs = self.features_ph
            s_t = self._slicing_init(inputs)  # Central slice
            c_t = self._context_net_init(inputs)  # Context vector
            concat_inputs = tf.concat([s_t, c_t], 1)
            h_t = self._fc_net_init(concat_inputs)  # Fully Connected baseline

            # Final Classification
            with tf.variable_scope("output"):
                logits = tf.layers.dense(inputs=h_t, units=1, activation=None, name="logits", reuse=False)
                prob = tf.sigmoid(logits, name="prob")
        return logits, prob

    def _slicing_init(self, inputs, reuse=False):
        with tf.variable_scope("slicing"):
            central_slice = inputs[:, :, int(self.context_size/2), :]
            central_slice = tf.expand_dims(central_slice, 2)  # Make it 4D tensor
            s_t = tf.layers.conv2d(inputs=central_slice, filters=1, kernel_size=1, strides=1, activation=tf.nn.relu,
                                   padding="same", name="proj", reuse=reuse)
            dim = np.prod(s_t.get_shape().as_list()[1:])
            s_t_flat = tf.reshape(s_t, shape=(-1, dim))
        return s_t_flat

    def _context_net_init(self, inputs, reuse=False):
        # Implement basic CNN from my notes (for now is just flattening spectrogram)
        with tf.variable_scope("context_net"):
            dim = np.prod(inputs.get_shape().as_list()[1:])
            context_net = tf.reshape(inputs, shape=(-1, dim))
        return context_net

    def _fc_net_init(self, inputs, reuse=False):
        with tf.variable_scope("fc_net"):
            fc_1 = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu, name="fc_1", reuse=reuse)
            fc_2 = tf.layers.dense(inputs=fc_1, units=512, activation=tf.nn.relu, name="fc_2", reuse=reuse)
        return fc_2

    def _loss_init(self):
        # We need to add L2 regularization and weights for balancing
        with tf.name_scope("loss"):
            byexample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.labels_ph)
            loss = tf.reduce_mean(byexample_loss)
        return loss

    def _optimizer_init(self):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()
            train_step = optimizer.minimize(self.loss)
        return train_step



