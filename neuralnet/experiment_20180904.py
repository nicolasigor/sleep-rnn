from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
import numpy as np
import itertools
import os
from models import lstm_model_v0, blstm_model_v0


if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    model_params = {
        "fs": dataset.get_fs()
    }

    # Build experiments
    max_it = 20000
    stat_every = 100

    log_transform_list = [False, True]
    cwt_bn_list = [False, True]
    fb_1 = np.array([1.5])
    fb_2 = np.array([0.5, 1.0, 1.5, 2.0])
    fb_array_list = [fb_1, fb_2]
    lstm_units_list = [128, 256]

    train_params_list = []
    for fb_array, log_transform, cwt_bn, lstm_units in itertools.product(fb_array_list,
                                                                         log_transform_list,
                                                                         cwt_bn_list,
                                                                         lstm_units_list):
        train_params = {
            "log_transform": log_transform,
            "cwt_bn": cwt_bn,
            "fb_array": fb_array,
            "lstm_units": lstm_units
        }
        train_params_list.append(train_params)

    experiment_name = "20180904_experiment"
    experiment_path = 'results/' + experiment_name + '/'

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    file = open(experiment_path + "README.txt", "w")
    file.write("Simple baseline of a 2 layers lstm model and sigmoid output, without CNN at input.\n")
    file.write("Without regularization on layers after CWT. Just to see the effects of architectural variations.\n")
    file.write("CWT is Cmorlet with fb eithet 1.5 or [0.5, 1.0, 1.5, 2.0].")
    file.close()

    # LSTM model, unidirectional, 2-layers
    model_fn = lstm_model_v0
    for train_params in train_params_list:
        str_params = "fb" + str(len(train_params["fb_array"]))
        str_params = str_params + "_log" + str(int(train_params["log_transform"]))
        str_params = str_params + "_bn" + str(int(train_params["cwt_bn"]))
        str_params = str_params + "_units" + str(train_params["lstm_units"])
        str_params = str_params + "_dir1"
        model_path = experiment_path + str_params + '/'
        # Train detector
        # print(model_path)
        tf.set_random_seed(0)
        detector = SpindleDetectorLSTM(model_params, model_path=model_path, model_fn=model_fn)
        detector.train(dataset, max_it, stat_every, train_params)
        del detector
        tf.reset_default_graph()

    # LSTM model, bidirectional, 2-layers
    model_fn = blstm_model_v0
    for train_params in train_params_list:
        str_params = "fb" + str(len(train_params["fb_array"]))
        str_params = str_params + "_log" + str(int(train_params["log_transform"]))
        str_params = str_params + "_bn" + str(int(train_params["cwt_bn"]))
        str_params = str_params + "_units" + str(train_params["lstm_units"])
        str_params = str_params + "_dir1"
        model_path = experiment_path + str_params + '/'
        # Train detector
        # print(model_path)
        tf.set_random_seed(0)
        detector = SpindleDetectorLSTM(model_params, model_path=model_path, model_fn=model_fn)
        detector.train(dataset, max_it, stat_every, train_params)
        del detector
        tf.reset_default_graph()
