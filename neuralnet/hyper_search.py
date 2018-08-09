from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
from models import lstm_model, blstm_model

if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    model_params = {
        "fs": dataset.get_fs()
    }

    # Build experiments
    max_it = 15000
    stat_every = 100
    n_retries = 3
    batch_size = 32
    lr_list = [1e-2, 1e-3, 1e-4, 1e-5]
    pos_weight_list = [0.5, 0.6, 0.7]

    train_params_list = []
    for lr in lr_list:
        for pos_weight in pos_weight_list:
            train_params = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "class_weights": [1-pos_weight, pos_weight]
            }
            train_params_list.append(train_params)

    # LSTM model, unidirectional, 2-layers, 256
    model_fn = lstm_model
    for train_params in train_params_list:
        for i in range(n_retries):
            str_params = "batch" + str(train_params["batch_size"])
            str_params = str_params + "pos" + str(train_params["class_weights"][1])
            str_params = str_params + "lr" + str(train_params["learning_rate"])
            model_path = 'results/lstm_2l_256u/' + str_params + '/run' + str(i+1) + '/'
            # Train detector
            detector = SpindleDetectorLSTM(model_params, model_path=model_path, model_fn=model_fn)
            detector.train(dataset, max_it, stat_every, train_params)
            del detector
            tf.reset_default_graph()

    # LSTM model, bidirectional, 2-layers, 256
    model_fn = blstm_model
    for train_params in train_params_list:
        for i in range(n_retries):
            str_params = "batch" + str(train_params["batch_size"])
            str_params = str_params + "pos" + str(train_params["class_weights"][1])
            str_params = str_params + "lr" + str(train_params["learning_rate"])
            model_path = 'results/blstm_2l_256u/' + str_params + '/run' + str(i + 1) + '/'
            # Train detector
            detector = SpindleDetectorLSTM(model_params, model_path=model_path, model_fn=model_fn)
            detector.train(dataset, max_it, stat_every, train_params)
            del detector
            tf.reset_default_graph()
