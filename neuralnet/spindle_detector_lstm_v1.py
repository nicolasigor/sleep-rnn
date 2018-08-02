from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import datetime

from model_lstm import model_lstm_v1


class SpindleDetectorLSTM(object):

    def __init__(self, params):
        # Load params, and fill with defaults when applicable
        if "fs" not in params:
            raise Exception("Please provide 'fs' in params.")
        if "page_sec" not in params:
            raise Exception("Please provide 'page_sec' in params.")
        self.p = params
        self._fix_to_default()

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = 'lstm'
        self.model_path = 'results/' + self.name + '_' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summ/'

    def train(self, dataset, max_it, stat_every, train_params):
        train_params = self._fix_to_default_train(train_params)
        with tf.device('/cpu:0'):
            feats_train, labels_train = dataset.get_augmented_numpy_subset(
                subset_name="train", mark_mode=1, border_sec=self.p["border_sec"])
            feats_val, labels_val = dataset.get_augmented_numpy_subset(
                subset_name="val", mark_mode=1, border_sec=self.p["border_sec"])
            iterator_train, feats_train_ph, labels_train_ph = self._iterator_init(feats_train, labels_train)
            iterator_val, feats_val_ph, labels_val_ph = self._iterator_init(feats_val, labels_val)




        start_time = time.time()
        print("Beginning training of " + self.name)


    def _fix_to_default(self):
        if "fc_array" not in self.p:
            self.p["fc_array"] = np.array([1, 1, 1, 1])
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
        if "context_sec" not in self.p:
            self.p["local_context_sec"] = 1

        if "time_stride" not in self.p:
            self.p["time_stride"] = 10
        if "context_stride" not in self.p:
            self.p["local_context_stride"] = 2


    def _fix_to_default_train(self, train_params):
        if "dropout_rate" not in train_params:
            train_params["dropout_rate"] = None
        if "batch_size" not in train_params:
            train_params["batch_size"] = 32
        if "learning_rate" not in train_params:
            train_params["learning_rate"] = None
        return train_params
