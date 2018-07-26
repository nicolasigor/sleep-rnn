from __future__ import division
import tensorflow as tf
import numpy as np
import time
import datetime

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


class DetectorFC(object):

    def __init__(self, params):
        # Training parameters
        self.dropout_rate = 0.3
        self.pos_weight = 3
        self.learning_rate = 0.0001
        self.batch_size = 256

        # General parameters
        self.context = params['context']
        self.factor_border = params['factor_border']
        self.mark_smooth = params['mark_smooth']
        self.fs = params["fs"]

        # CWT parameters
        self.cwt_stride = 2
        self.fc = 1
        self.fb_array = np.array([0.5, 1, 1.5, 2])
        self.n_scales = 32
        self.upper_freq = 40
        self.lower_freq = 2
        # Generate initial and last scale
        s_0 = self.fs / self.upper_freq
        s_n = self.fs / self.lower_freq
        # Generate the array of scales
        base = np.power(s_n / s_0, 1 / (self.n_scales - 1))
        self.scales = s_0 * np.power(base, range(self.n_scales))
        # Generate kernels
        self.kernels_morlet = []
        for j in range(self.fb_array.shape[0]):
            fb = self.fb_array[j]
            fb_kernels_morlet = []
            for i in range(self.n_scales):
                this_scale = self.scales[i]
                one_side = int(this_scale * np.sqrt(10 * fb))
                this_kernel_size = 2 * one_side + 1
                this_k = np.arange(this_kernel_size, dtype=np.float32) - one_side
                this_kernel = np.exp(-((this_k / this_scale) ** 2) / fb) / np.sqrt(np.pi * fb * this_scale)
                this_kernel_real = this_kernel * np.cos(2 * np.pi * self.fc * this_k / this_scale)
                this_kernel_imag = this_kernel * np.sin(2 * np.pi * self.fc * this_k / this_scale)
                useful_shape = (1,) + this_kernel_real.shape + (1, 1)
                this_kernel_real = np.reshape(this_kernel_real, useful_shape)
                this_kernel_imag = np.reshape(this_kernel_imag, useful_shape)
                fb_kernels_morlet.append((this_kernel_real, this_kernel_imag))
            self.kernels_morlet.append(fb_kernels_morlet)

        # Some static values
        self.segment_size = int((self.factor_border + 1) * self.context * self.fs)
        self.context_size = int(self.context * self.fs / self.cwt_stride)
        self.context_start = int(self.factor_border * self.context * self.fs / (2*self.cwt_stride))
        self.context_end = self.context_start + self.context_size

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_name = 'detector_fc_v1'
        self.model_path = 'results/' + self.model_name + ' ' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summaries/'

        # Model initialization
        with tf.name_scope("input"):
            self.features_ph = tf.placeholder(shape=[None, 1, self.segment_size, 1],
                                              dtype=tf.float32, name="features")
            self.labels_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="labels")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.inputs_cwt, self.logits, self.prob = self._model_init()
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
                                                  dataset="TRAIN")
            _, train_loss, summ = self.sess.run([self.train_step, self.loss, merged],
                                                feed_dict={self.features_ph: features,
                                                           self.labels_ph: labels,
                                                           self.is_training: True})

            if it % stat_every == 0:
                # Training stat
                self.train_writer.add_summary(summ, it)
                # cwt_image = self.sess.run(self.inputs_cwt,
                #                           feed_dict={self.features_ph: features,
                #                                      self.labels_ph: labels})
                # Validation stat
                features, labels = dataset.next_batch(batch_size=self.batch_size,
                                                      segment_size=self.segment_size,
                                                      mark_smooth=self.mark_smooth,
                                                      dataset="VAL")
                val_loss, summ = self.sess.run([self.loss, merged],
                                               feed_dict={self.features_ph: features,
                                                          self.labels_ph: labels,
                                                          self.is_training: False})
                self.val_writer.add_summary(summ, it)

                time_usage = time.time() - start_time
                print("Iteration %i/%i: train loss %f - val loss %f - time %f s"
                      % (it, max_it, train_loss, val_loss, time_usage), flush=True)
                print("Iteration %i/%i: train loss %f - val loss %f - time %f s"
                      % (it, max_it, train_loss, val_loss, time_usage),
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
            inputs_cwt = self._cwt_init(inputs)  # Convert signal into CWT

            inputs_cwt_bn = tf.layers.batch_normalization(inputs=inputs_cwt, training=self.is_training,
                                                          reuse=reuse, name="bn_1")

            s_t = self._slicing_init(inputs_cwt_bn)  # Central slice
            c_t = self._context_net_init(inputs_cwt_bn)  # Context vector

            concat_inputs = tf.concat([s_t, c_t], 1)
            h_t = self._fc_net_init(concat_inputs)  # Fully Connected baseline

            # Final Classification
            with tf.variable_scope("output"):
                logits = tf.layers.dense(inputs=h_t, units=1, activation=None, name="logits", reuse=reuse)
                prob = tf.sigmoid(logits, name="prob")

        return inputs_cwt, logits, prob

    def _cwt_init(self, inputs):
        with tf.name_scope("cwt"):
            cwt_list = []
            for j in range(self.fb_array.shape[0]):
                with tf.name_scope("fb_"+str(j)):
                    fb_cwt_list = []
                    for i in range(self.n_scales):
                        this_kernel_tuple = self.kernels_morlet[j][i]
                        power = self._apply_complex_kernel(this_kernel_tuple, inputs)
                        # Spectrum flattening
                        power = power * (self.fs / self.scales[i])
                        fb_cwt_list.append(power)
                    single_scalogram = tf.concat(fb_cwt_list, 1)
                    cwt_list.append(single_scalogram)
            cwt_op = tf.concat(cwt_list, -1)
        return cwt_op

    def _apply_complex_kernel(self, kernel_tuple, input_signal):
        kernel_real = tf.constant(kernel_tuple[0])
        kernel_imag = tf.constant(kernel_tuple[1])
        out_real = tf.nn.conv2d(input=input_signal, filter=kernel_real,
                                strides=[1, 1, self.cwt_stride, 1], padding="SAME")
        out_imag = tf.nn.conv2d(input=input_signal, filter=kernel_imag,
                                strides=[1, 1, self.cwt_stride, 1], padding="SAME")
        start = self.context_start
        end = self.context_end
        out_real_context = out_real[:, :, start:end, :]
        out_imag_context = out_imag[:, :, start:end, :]
        out_abs = tf.sqrt(tf.square(out_real_context) + tf.square(out_imag_context))
        return out_abs

    def _slicing_init(self, inputs, reuse=False):
        with tf.variable_scope("slicing"):
            central_pos = int(self.context_size/2)
            central_slice = inputs[:, :, central_pos:(central_pos+1), :]
            s_t = tf.layers.conv2d(inputs=central_slice, filters=1, kernel_size=1, activation=tf.nn.relu,
                                   padding="same", name="c_1x1", reuse=reuse)
            dim = np.prod(s_t.get_shape().as_list()[1:])
            s_t_flat = tf.reshape(s_t, shape=(-1, dim))
        return s_t_flat

    def _context_net_init(self, inputs, reuse=False):
        with tf.variable_scope("context_net"):
            # First convolutional block
            c_1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, activation=tf.nn.relu,
                                   padding="same", name="c_1", reuse=reuse)
            p_1 = tf.layers.max_pooling2d(inputs=c_1, pool_size=2, strides=2)

            # Second convolutional block
            p_1_bn = tf.layers.batch_normalization(inputs=p_1, training=self.is_training,
                                                   name="bn_2",reuse=reuse)
            c_2 = tf.layers.conv2d(inputs=p_1_bn, filters=32, kernel_size=3, activation=tf.nn.relu,
                                   padding="same", name="c_2", reuse=reuse)
            p_2 = tf.layers.max_pooling2d(inputs=c_2, pool_size=2, strides=2)

            # Third convolutional block
            p_2_bn = tf.layers.batch_normalization(inputs=p_2, training=self.is_training,
                                                   name="bn_3a", reuse=reuse)
            c_3a = tf.layers.conv2d(inputs=p_2_bn, filters=64, kernel_size=3, activation=tf.nn.relu,
                                    padding="same", name="c_3a", reuse=reuse)
            c_3a_bn = tf.layers.batch_normalization(inputs=c_3a, training=self.is_training,
                                                    name="bn_3b", reuse=reuse)
            c_3b = tf.layers.conv2d(inputs=c_3a_bn, filters=64, kernel_size=3, activation=tf.nn.relu,
                                    padding="same", name="c_3b", reuse=reuse)
            p_3 = tf.layers.max_pooling2d(inputs=c_3b, pool_size=2, strides=2)

            # Fourth convolutional block
            p_3_bn = tf.layers.batch_normalization(inputs=p_3, training=self.is_training,
                                                   name="bn_4a", reuse=reuse)
            c_4a = tf.layers.conv2d(inputs=p_3_bn, filters=128, kernel_size=3, activation=tf.nn.relu,
                                    padding="same", name="c_4a", reuse=reuse)
            c_4a_bn = tf.layers.batch_normalization(inputs=c_4a, training=self.is_training,
                                                    name="bn_4b", reuse=reuse)
            c_4b = tf.layers.conv2d(inputs=c_4a_bn, filters=128, kernel_size=3, activation=tf.nn.relu,
                                    padding="same", name="c_4b", reuse=reuse)
            p_4 = tf.layers.average_pooling2d(inputs=c_4b, pool_size=4, strides=4)

            # Flattening
            dim = np.prod(p_4.get_shape().as_list()[1:])
            c_t_flat = tf.reshape(p_4, shape=(-1, dim))
        return c_t_flat

    def _fc_net_init(self, inputs, reuse=False):
        with tf.variable_scope("fc_net"):
            inputs_dp = tf.layers.dropout(inputs=inputs, rate=self.dropout_rate, training=self.is_training, name="dp_1")
            fc_1 = tf.layers.dense(inputs=inputs_dp, units=512, activation=tf.nn.relu, name="fc_1", reuse=reuse)
            fc_1_dp = tf.layers.dropout(inputs=fc_1, rate=self.dropout_rate, training=self.is_training, name="dp_2")
            fc_2 = tf.layers.dense(inputs=fc_1_dp, units=256, activation=tf.nn.relu, name="fc_2", reuse=reuse)
        return fc_2

    def _loss_init(self):
        # We need to add L2 regularization and weights for balancing
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
        return loss

    def _optimizer_init(self):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For BN
            with tf.control_dependencies(update_ops):
                train_step = optimizer.minimize(self.loss)
        return train_step
