from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM

if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs()
    }
    detector = SpindleDetectorLSTM(model_params)

    # Train detector
    train_params = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "class_weights": [0.25, 0.75]
    }
    max_it = 100000
    stat_every = 500
    detector.train(dataset, max_it, stat_every, train_params)
