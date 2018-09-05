from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
from models import lstm_model_v0
import numpy as np

if __name__ == "__main__":
    tf.set_random_seed(0)
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs(),
        "fb_array": np.array([1.5]),  # np.array([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0]),
        "border_sec": 5
    }
    # Train detector
    train_params = {
        "learning_rate": 1e-4,
        "drop_rate": 0.0
    }
    max_it = 10000
    stat_every = 50

    # Unidirectional
    detector = SpindleDetectorLSTM(model_params, model_fn=lstm_model_v0)
    detector.train(dataset, max_it, stat_every, train_params)

    # del detector
    # tf.reset_default_graph()
    # tf.set_random_seed(0)
    # Bidirectional
    # detector = SpindleDetectorLSTM(model_params, model_fn=blstm_model_v2)
    # detector.train(dataset, max_it, stat_every, train_params)
