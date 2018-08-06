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
        "learning_rate": 1e-3,
        "batch_size": 32
    }
    max_it = 100
    stat_every = 10
    detector.train(dataset, max_it, stat_every, train_params)
