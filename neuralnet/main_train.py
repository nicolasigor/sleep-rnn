from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
from models import lstm_model_v2


if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs(),
        "fb_array": np.array([0.5, 1, 1.5, 2])
    }
    # Train detector
    train_params = {
        "learning_rate": 1e-3,
        "drop_rate": 0.0
    }
    max_it = 5000
    stat_every = 50

    # Unidirectional
    detector = SpindleDetectorLSTM(model_params, model_fn=lstm_model_v2)
    detector.train(dataset, max_it, stat_every, train_params)

    # del detector
    # tf.reset_default_graph()

    # Bidirectional
    # detector = SpindleDetectorLSTM(model_params, model_fn=blstm_model_v3)
    # detector.train(dataset, max_it, stat_every, train_params)
