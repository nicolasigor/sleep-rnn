from sleep_data import SleepDataMASS
from spindle_detector_lstm import SpindleDetectorLSTM
import tensorflow as tf
# from models import lstm_model_v0, blstm_model_v0
# from models import lstm_model_v0_cnn, blstm_model_v0_cnn
from models import model_for_ppt
import numpy as np

if __name__ == "__main__":
    # Sleep data
    dataset = SleepDataMASS(load_from_checkpoint=True)

    # Instance of detector
    model_params = {
        "fs": dataset.get_fs(),
        "fb_array": np.array([0.5, 1.0, 1.5, 2.0]),  # np.array([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0]),
        "border_sec": 5
    }

    # Train detector
    train_params = {
        "init_learning_rate": 1e-4,
        "drop_rate": 0.0,
        "time_stride": 10,
    }

    ckpt_path = "results/sequential_20180906-013602/checkpoints"

    detector = SpindleDetectorLSTM(model_params, model_fn=model_for_ppt)

    # Predict validation data
    feats_val, labels_val = dataset.get_augmented_numpy_subset(
        subset_name="val", mark_mode=1, border_sec=model_params["border_sec"])
    # predictions_val = detector.predict(train_params, ckpt_path, feats_val)
    # print(labels_val[:, 3000:7000:10].shape)
    np.savetxt("expert1_val.csv", labels_val[:, 3000:7000:10])
    # print(feats_val.shape, labels_val.shape, predictions_val.shape)
    # np.savetxt("predictions_val_right.csv", predictions_val[:, :, 1])
    
    # Predict test data
    feats_test, labels_test = dataset.get_augmented_numpy_subset(
        subset_name="test", mark_mode=1, border_sec=model_params["border_sec"])
    np.savetxt("expert1_test.csv", labels_test[:, 3000:7000:10])
    # predictions_test = detector.predict(train_params, ckpt_path, feats_test)
    
    # print(feats_test.shape, labels_test.shape, predictions_test.shape)
    # np.savetxt("predictions_test_right.csv", predictions_test[:, :, 1])

