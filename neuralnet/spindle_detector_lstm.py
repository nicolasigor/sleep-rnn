from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import datetime


class SpindleDetectorLSTM(object):

    def __init__(self, model_params, model_fn, model_path=None):
        # Load params, and fill with defaults when applicable
        if "fs" not in model_params:
            raise Exception("Please provide 'fs' in params.")
        self.model_fn = model_fn
        self.p = model_params
        self._default_model_params()

        # Directories
        self.name = 'sequential'
        if model_path is None:
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.model_path = 'results/' + self.name + '_' + date + '/'
        else:
            self.model_path = model_path
        self.ckpt_path = None

    def train(self, dataset, n_epochs, train_params=None):
        # Reset everything
        tf.reset_default_graph()

        # Combine params
        if train_params is not None:
            self.p.update(train_params)
        self._default_train_params()

        # Read numpy data
        feats_train, labels_train = dataset.get_augmented_numpy_subset(
            subset_name="train", mark_mode=1, border_sec=self.p["border_sec"])
        feats_val, labels_val = dataset.get_augmented_numpy_subset(
            subset_name="val", mark_mode=1, border_sec=self.p["border_sec"])

        # Build input pipeline
        with tf.name_scope("input_ph"):
            feats_train_ph = tf.placeholder(dtype=tf.float32, shape=feats_train.shape, name="feats_train_ph")
            labels_train_ph = tf.placeholder(dtype=tf.int32, shape=labels_train.shape, name="labels_train_ph")
            feats_val_ph = tf.placeholder(dtype=tf.float32, shape=feats_val.shape, name="feats_val_ph")
            labels_val_ph = tf.placeholder(dtype=tf.int32, shape=labels_val.shape, name="labels_val_ph")
            training_ph = tf.placeholder(tf.bool, name="training_ph")
        iterator, handle_ph, iter_train, iter_val = self._iter_training_init(
            feats_train_ph, labels_train_ph, feats_val_ph, labels_val_ph, self.p["batch_size"])

        # Build model graph
        loss, train_step, metrics, feats, predictions = self._build_training_graph(
            iterator, training_ph, self.p["learning_rate"], self.p["class_weights"])

        # Session, savers and writers
        self.ckpt_path = self.model_path + 'checkpoints/model'
        tb_path = self.model_path + 'tb_summ/'
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=100)
        # train_writer = tf.summary.FileWriter(tb_path + 'train', sess.graph)
        # val_writer = tf.summary.FileWriter(tb_path + 'val')

        # Initialization of variables
        sess.run(tf.global_variables_initializer())
        # merged = tf.summary.merge_all()

        # Initialization of iterators
        # sess.run(iter_train.initializer, feed_dict={feats_train_ph: feats_train, labels_train_ph: labels_train})
        # sess.run(iter_val.initializer, feed_dict={feats_val_ph: feats_val, labels_val_ph: labels_val})
        # train_handle, val_handle = sess.run([iter_train.string_handle(), iter_val.string_handle()])

        start_time = time.time()
        print("Beginning training of " + self.name + " at " + self.model_path)

        ema_train = 0.8
        train_loss_ema = 0
        train_f1_ema = 0

        # Epoch 0
        # Initialization of iterators
        sess.run(iter_train.initializer, feed_dict={feats_train_ph: feats_train, labels_train_ph: labels_train})
        sess.run(iter_val.initializer, feed_dict={feats_val_ph: feats_val, labels_val_ph: labels_val})
        train_handle, val_handle = sess.run([iter_train.string_handle(), iter_val.string_handle()])
        while True:
            try:
                feed_dict = {handle_ph: train_handle, training_ph: False}
                train_loss, train_metrics = sess.run([loss, metrics], feed_dict=feed_dict)
                train_loss_ema = train_loss_ema + ema_train * (train_loss - train_loss_ema)
                train_f1_ema = train_f1_ema + ema_train * (train_metrics["bs_f1_score"] - train_f1_ema)
            except tf.errors.OutOfRangeError:
                # train_writer.add_summary(train_summ, i)
                break

        val_loss_total = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0
        count = 0
        while True:
            try:
                count += 1
                feed_dict = {handle_ph: val_handle, training_ph: False}
                val_loss, val_metrics = sess.run([loss, metrics], feed_dict=feed_dict)
                val_loss_total = val_loss_total + (val_loss - val_loss_total) / count
                val_tp += val_metrics["tp"]
                val_fp += val_metrics["fp"]
                val_fn += val_metrics["fn"]
            except tf.errors.OutOfRangeError:
                # val_writer.add_summary(val_summ, i)
                val_loss = val_loss_total
                val_precision = val_tp / (val_tp + val_fp)
                val_recall = val_tp / (val_tp + val_fn)
                val_f1 = 2 * val_precision * val_recall / (val_recall + val_precision)
                break
        # [Perform end-of-epoch calculations here.]
        elapsed_time = time.time() - start_time
        print("Epoch %1.3i/%1.3i -- train_loss %1.6f f1 %1.4f -- val loss %1.6f f1 %1.4f -- elap time %f [s]" %
              (0, n_epochs, train_loss_ema, train_f1_ema, val_loss, val_f1, elapsed_time))

        for i in range(1, n_epochs+1):
            # Initialization of iterators
            sess.run(iter_train.initializer, feed_dict={feats_train_ph: feats_train, labels_train_ph: labels_train})
            sess.run(iter_val.initializer, feed_dict={feats_val_ph: feats_val, labels_val_ph: labels_val})
            train_handle, val_handle = sess.run([iter_train.string_handle(), iter_val.string_handle()])
            while True:
                try:
                    feed_dict = {handle_ph: train_handle, training_ph: True}
                    _, train_loss, train_metrics = sess.run([train_step, loss, metrics], feed_dict=feed_dict)
                    train_loss_ema = train_loss_ema + ema_train*(train_loss - train_loss_ema)
                    train_f1_ema = train_f1_ema + ema_train*(train_metrics["bs_f1_score"] - train_f1_ema)
                except tf.errors.OutOfRangeError:
                    # train_writer.add_summary(train_summ, i)
                    break

            val_loss_total = 0
            val_tp = 0
            val_fp = 0
            val_fn = 0
            count = 0
            while True:
                try:
                    count += 1
                    feed_dict = {handle_ph: val_handle, training_ph: False}
                    val_loss, val_metrics = sess.run([loss, metrics], feed_dict=feed_dict)
                    val_loss_total = val_loss_total + (val_loss - val_loss_total)/count
                    val_tp += val_metrics["tp"]
                    val_fp += val_metrics["fp"]
                    val_fn += val_metrics["fn"]
                except tf.errors.OutOfRangeError:
                    # val_writer.add_summary(val_summ, i)
                    val_loss = val_loss_total
                    val_precision = val_tp/(val_tp+val_fp)
                    val_recall = val_tp/(val_tp+val_fn)
                    val_f1 = 2*val_precision*val_recall/(val_recall + val_precision)
                    break
            # [Perform end-of-epoch calculations here.]
            elapsed_time = time.time() - start_time
            print("Epoch %1.3i/%1.3i -- train_loss %1.6f f1 %1.4f -- val loss %1.6f f1 %1.4f -- elap time %f [s]" %
                  (i, n_epochs, train_loss_ema, train_f1_ema, val_loss, val_f1, elapsed_time))
            if i % 5 == 0 and i != n_epochs:
                save_path = saver.save(sess, self.ckpt_path, global_step=i)
                print("Model saved to: %s" % save_path)
        '''
        for it in range(1, max_it+1):
            feed_dict = {handle_ph: train_handle, training_ph: True}
            # sess.run(train_step, feed_dict=feed_dict)
            _, train_loss, train_summ, train_metrics = sess.run([train_step, loss, merged, metrics], feed_dict=feed_dict)
            if it % stat_every == 0:
                # feed_dict = {handle_ph: train_handle, training_ph: False}
                # train_loss, train_summ, train_metrics = sess.run([loss, merged, metrics], feed_dict=feed_dict)
                feed_dict = {handle_ph: val_handle, training_ph: False}
                val_loss, val_summ, val_metrics = sess.run([loss, merged, metrics], feed_dict=feed_dict)
                train_writer.add_summary(train_summ, it)
                val_writer.add_summary(val_summ, it)
                elapsed_time = time.time() - start_time
                print("Iter %i/%i -- train loss %1.6f f1 %1.4f -- val loss %1.6f f1 %1.4f -- elap time %f [s]"
                      % (it, max_it, train_loss, train_metrics["bs_f1_score"], val_loss, val_metrics["bs_f1_score"], elapsed_time))
        
        '''
        save_path = saver.save(sess, self.ckpt_path, global_step=n_epochs)
        print("Model saved to: %s" % save_path)

        # PREDICT ON VAL AND TEST SET
        feats_val, labels_val = dataset.get_augmented_numpy_subset(
            subset_name="val", mark_mode=1, border_sec=self.p["border_sec"])
        feats_test, labels_test = dataset.get_augmented_numpy_subset(
            subset_name="test", mark_mode=1, border_sec=self.p["border_sec"])

        # Reset everything
        tf.reset_default_graph()

    def _default_model_params(self):
        if "fb_array" not in self.p:
            self.p["fb_array"] = np.array([0.5, 1, 1.5, 2])
        if "lower_freq" not in self.p:
            self.p["lower_freq"] = 3
        if "upper_freq" not in self.p:
            self.p["upper_freq"] = 40
        if "n_scales" not in self.p:
            self.p["n_scales"] = 32

        if "border_sec" not in self.p:
            self.p["border_sec"] = 1
        if "local_context_sec" not in self.p:
            self.p["local_context_sec"] = 1
        if "page_sec" not in self.p:
            self.p["page_sec"] = 20

        if "time_stride" not in self.p:
            self.p["time_stride"] = 8

        self.p["border_size"] = int(self.p["border_sec"] * self.p["fs"])
        self.p["page_size"] = int(self.p["page_sec"] * self.p["fs"])
        self.p["local_context_size"] = int(self.p["local_context_sec"] * self.p["fs"])
        self.p["crop_size"] = self.p["page_size"] + 2*self.p["border_size"]

    def _default_train_params(self):
        if "batch_size" not in self.p:
            self.p["batch_size"] = 32
        if "learning_rate" not in self.p:
            self.p["learning_rate"] = 1e-4
        if "class_weights" not in self.p:
            self.p["class_weights"] = [1, 1]
        if "drop_rate" not in self.p:
            self.p["drop_rate"] = 0.0
        # if "clip_gradients" not in self.p:
        #     self.p["clip_gradients"] = False
        #     self.p["clip_norm"] = None
        # elif self.p["clip_gradients"] and ("clip_norm" not in self.p):
        #     self.p["clip_norm"] = 5

    def _iter_training_init(self, feats_train_ph, labels_train_ph, feats_val_ph, labels_val_ph, batch_size):
        with tf.device('/cpu:0'):
            with tf.name_scope("iterators"):
                dataset_train = self._dataset_init(feats_train_ph, labels_train_ph, batch_size)
                dataset_val = self._dataset_init(feats_val_ph, labels_val_ph, batch_size)
                iter_train = dataset_train.make_initializable_iterator()
                iter_val = dataset_val.make_initializable_iterator()
                handle_ph = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                    handle_ph, dataset_train.output_types, dataset_train.output_shapes)
        return iterator, handle_ph, iter_train, iter_val

    def _dataset_init(self, feats_ph, labels_ph, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((feats_ph, labels_ph))
        # dataset = dataset.repeat()
        dataset = dataset.map(self._random_crop_fn)
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=2)
        return dataset

    def _random_crop_fn(self, feat, label):
        crop_size = self.p["crop_size"]
        time_stride = self.p["time_stride"]
        border_size = self.p["border_size"]
        # Random crop
        label_cast = tf.cast(label, dtype=tf.float32)
        stack = tf.stack([feat, label_cast], axis=0)
        stack_crop = tf.random_crop(stack, [2, crop_size])
        feat = stack_crop[0, :]
        # Throw first and last second for labels, skipping steps
        label_cast = stack_crop[1, border_size:-border_size:time_stride]
        label = tf.cast(label_cast, dtype=tf.int32)
        return feat, label

    def _central_crop_fn(self, feat):
        # Central crop
        print("CROP",self.p["crop_size"])
        out = int(self.p["page_size"]/2)
        print("out:", out)
        print("feat", feat)
        feat = feat[out:-out]
        print("feat v2", feat)
        return feat

    def _build_training_graph(self, iterator, training_ph, lr, class_weights):
        # Input
        feats, labels = iterator.get_next()
        # tf.summary.histogram("labels", labels)
        # Model
        logits, predictions = self.model_fn(feats, self.p, training=training_ph)
        # Optimization ops
        loss = self._loss_init(logits, labels, class_weights)
        train_step = self._optimizer_init(loss, lr)
        # Metrics
        metrics = self._batch_metrics_init(predictions, labels)
        return loss, train_step, metrics, feats, predictions

    def _build_evaluation_graph(self, iterator):
        # TODO: evaluation graph
        pass

    def _build_prediction_graph(self, iterator):
        # TODO: prediction graph
        # Input
        feats = iterator.get_next()
        # Model
        _, predictions = self.model_fn(feats, self.p, training=False)
        return predictions

    def _loss_init(self, logits, labels, class_weights):
        with tf.name_scope("loss"):
            # Shape of labels is [batch, timesteps]. Shape of logits is [batch, timesteps, 2]
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
            tf.summary.scalar('loss', loss)
        return loss

    def _optimizer_init(self, loss, lr):
        # TODO: track histogram of weights and study l2 regularization
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            gvs = optimizer.compute_gradients(loss)
            # if self.p["clip_gradients"]:
            #     gvs = [(tf.clip_by_norm(gv[0], self.p["clip_norm"]), gv[1]) for gv in gvs]
            #
            # # Histogram of gradients norm
            # with tf.name_scope("grads_summ"):
            #     grad_norm_list = [tf.sqrt(tf.reduce_sum(gv[0] ** 2)) for gv in gvs]
            #     grad_norm_stacked = tf.stack(grad_norm_list)
            #     tf.summary.histogram('grad_norm', grad_norm_stacked)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For BN
            with tf.control_dependencies(update_ops):
                # train_step = optimizer.minimize(loss)
                train_step = optimizer.apply_gradients(gvs)

        return train_step

    def _batch_metrics_init(self, predictions, labels):
        with tf.name_scope("metrics"):
            predictions_sparse = tf.argmax(predictions, axis=-1)

            labels_zero = tf.equal(labels, tf.zeros_like(labels))
            labels_one = tf.equal(labels, tf.ones_like(labels))
            predictions_zero = tf.equal(predictions_sparse, tf.zeros_like(predictions_sparse))
            predictions_one = tf.equal(predictions_sparse, tf.ones_like(predictions_sparse))

            tp = tf.reduce_sum(tf.cast(tf.logical_and(labels_one, predictions_one), "float"))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(labels_zero, predictions_one), "float"))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(labels_one, predictions_zero), "float"))

            # Edge case: no detections -> precision 1
            bs_precision = tf.cond(
                pred=tf.equal((tp+fp), 0),
                true_fn=lambda: 1.0,
                false_fn=lambda: tp/(tp+fp)
            )
            # Edge case: no marks -> recall 1
            bs_recall = tf.cond(
                pred=tf.equal((tp+fn), 0),
                true_fn=lambda: 1.0,
                false_fn=lambda: tp/(tp+fn)
            )
            # Edge case: precision and recall 0 -> f1 score 0
            bs_f1_score = tf.cond(
                pred=tf.equal((bs_precision + bs_recall), 0),
                true_fn=lambda: 0.0,
                false_fn=lambda: 2*bs_precision*bs_recall/(bs_precision + bs_recall)
            )
            tf.summary.scalar('bs_precision', bs_precision)
            tf.summary.scalar('bs_recall', bs_recall)
            tf.summary.scalar('bs_f1_score', bs_f1_score)
            metrics = {
                "bs_precision": bs_precision,
                "bs_recall": bs_recall,
                "bs_f1_score": bs_f1_score,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        return metrics

    def set_checkpoint(self, model_path):
        self.ckpt_path = model_path + 'checkpoints/model'

    def evaluate(self):
        # TODO: restore from checkpoint and evaluate
        pass

    def predict(self, train_params, ckpt_path, feats):
        # TODO: restore from checkpoint and predict
        # Reset everything
        tf.reset_default_graph()

        # Combine params
        if train_params is not None:
            self.p.update(train_params)
        self._default_train_params()

        # Build input pipeline
        with tf.device('/cpu:0'):
            with tf.name_scope("input_ph"):
                feats_ph = tf.placeholder(dtype=tf.float32, shape=feats.shape, name="feats_ph")
                dataset = tf.data.Dataset.from_tensor_slices(feats_ph)
                dataset = dataset.map(self._central_crop_fn)
                dataset = dataset.batch(batch_size=32)
                dataset = dataset.prefetch(buffer_size=2)
                iter_predict = dataset.make_initializable_iterator()

        # Build model graph
        prediction = self._build_prediction_graph(iter_predict)

        sess = tf.Session()
        saver = tf.train.Saver()
        print("Loading checkpoint at " + ckpt_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        # Initialization of iterators
        sess.run(iter_predict.initializer, feed_dict={feats_ph: feats})

        prediction_list = []
        while True:
            try:
                this_prediction = sess.run(prediction)
                print(this_prediction.shape)
            except tf.errors.OutOfRangeError:
                break
            prediction_list.append(this_prediction)
        prediction_total = np.concatenate(prediction_list)
        tf.reset_default_graph()
        return prediction_total
