from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
# from models import lstm_model_v0, blstm_model_v0
# from models import lstm_model_v0_cnn, blstm_model_v0_cnn
from models import model_for_ppt
import numpy as np

if __name__ == "__main__":
    # tf.set_random_seed(0)
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs(),
        "fb_array": np.array([0.5, 1.0, 1.5, 2.0]),  # np.array([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0]),
        "border_sec": 5
    }

    n_epochs = 100

    # Train detector
    train_params = {
        "learning_rate": 1e-4,
        "drop_rate": 0.0,
        "time_stride": 10,
        # "log_transform": False,
        # "cwt_bn": True,
        # "lstm_units": 256
    }

    # Unidirectional
    # detector = SpindleDetectorLSTM(model_params, model_fn=lstm_model_v0)
    # detector.train(dataset, n_epochs, train_params)

    # del detector
    # tf.reset_default_graph()
    # tf.set_random_seed(0)
    # Bidirectional
    # print("Without log transform")
    detector = SpindleDetectorLSTM(model_params, model_fn=model_for_ppt)
    detector.train(dataset, n_epochs, train_params)

    # del detector
    # tf.reset_default_graph()
    # tf.set_random_seed(0)
    #
    # # Train detector
    # train_params = {
    #     "learning_rate": 1e-4,
    #     "drop_rate": 0.0,
    #     "time_stride": 10,
    #     "log_transform": True,
    #     "cwt_bn": True,
    #     "lstm_units": 256
    # }
    #
    # # Bidirectional
    # print("With log transform")
    # detector = SpindleDetectorLSTM(model_params, model_fn=blstm_model_v0)
    # detector.train(dataset, n_epochs, train_params)
