from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
from models import lstm_model, blstm_model

if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs()
    }
    detector = SpindleDetectorLSTM(model_params, model_path='blstm', model_fn=blstm_model)

    # Train detector
    train_params = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "class_weights": [0.3, 0.7]
    }
    max_it = 1000
    stat_every = 100
    detector.train(dataset, max_it, stat_every, train_params)
