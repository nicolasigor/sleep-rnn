from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
from models import lstm_model_v2, blstm_model_v2


if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs()
    }
    # Train detector
    train_params = {
        "learning_rate": 1e-4,
        "drop_rate": 0.2
    }
    max_it = 100
    stat_every = 100

    # Unidirectional
    detector = SpindleDetectorLSTM(model_params, model_fn=lstm_model_v2)
    detector.train(dataset, max_it, stat_every, train_params)

    #del detector
    #tf.reset_default_graph()

    # Bidirectional
    #detector = SpindleDetectorLSTM(model_params, model_fn=blstm_model_v2)
    #detector.train(dataset, max_it, stat_every, train_params)
