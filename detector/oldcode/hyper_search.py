from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
from models import lstm_model_v3, blstm_model_v3


# TODO: usar las 4 gpus del cluster
if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    model_params = {
        "fs": dataset.get_fs()
    }

    # Build experiments
    max_it = 20000
    stat_every = 100
    n_retries = 2
    # batch_size = 32
    # pos_weight = 0.5
    lr_list = [1e-3, 1e-4, 1e-5]
    # drop_rate_list = [0.2, 0.35, 0.5]

    train_params_list = []
    for lr in lr_list:
        train_params = {
            "learning_rate": lr,
        }
        train_params_list.append(train_params)

    experiment_name = "20180813 Hyper search v3 CNN no context no drop"

    # LSTM model, unidirectional, 2-layers, 256
    model_fn = lstm_model_v3
    for train_params in train_params_list:
        for i in range(n_retries):
            # str_params = "batch" + str(train_params["batch_size"])
            str_params = "lr" + str(train_params["learning_rate"])
            # str_params = str_params + "dr" + str(train_params["drop_rate"])
            model_path = 'results/' + experiment_name + '/cnn_lstm_2l_256u/' + str_params + '/run' + str(i+1) + '/'
            # Train detector
            # print(model_path)
            detector = SpindleDetectorLSTM(model_params, model_path=model_path, model_fn=model_fn)
            detector.train(dataset, max_it, stat_every, train_params)
            del detector
            tf.reset_default_graph()

    # LSTM model, bidirectional, 2-layers, 256
    model_fn = blstm_model_v3
    for train_params in train_params_list:
        for i in range(n_retries):
            # str_params = "batch" + str(train_params["batch_size"])
            str_params = "lr" + str(train_params["learning_rate"])
            # str_params = str_params + "dr" + str(train_params["drop_rate"])
            model_path = 'results/' + experiment_name + '/cnn_blstm_2l_256u/' + str_params + '/run' + str(i + 1) + '/'
            # Train detector
            print(model_path)
            detector = SpindleDetectorLSTM(model_params, model_path=model_path, model_fn=model_fn)
            detector.train(dataset, max_it, stat_every, train_params)
            del detector
            tf.reset_default_graph()