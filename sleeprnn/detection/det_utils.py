import numpy as np
from sleeprnn.data import utils
from sleeprnn.detection.predicted_dataset import PredictedDataset


def transform_predicted_proba_to_adjusted_proba(predicted_proba, optimal_threshold, eps=1e-8):
    """
    Adjusts probability vector so that
    adjusted_proba > 0.5 is equivalent to predicted_proba > optimal_threshold

    :param predicted_proba: vector of predicted probabilities.
    :param optimal_threshold: optimal threshold for class assignment in predicted probabilities.
    :param eps: for numerical stability. Defaults to 1e-8.
    :return: the vector of adjusted probabilities.
    """
    # Prepare
    original_dtype = predicted_proba.dtype
    predicted_proba = predicted_proba.astype(np.float64)
    predicted_proba = np.clip(predicted_proba, a_min=eps, a_max=(1.0 - eps))
    # Compute
    logit_proba = np.log(predicted_proba / (1.0 - predicted_proba))
    bias_from_thr = -np.log(optimal_threshold / (1.0 - optimal_threshold))
    new_logit_proba = logit_proba + bias_from_thr
    adjusted_proba = 1.0 / (1.0 + np.exp(-new_logit_proba))
    # Go back to original dtype
    adjusted_proba = adjusted_proba.astype(original_dtype)
    return adjusted_proba


def transform_thr_for_adjusted_to_thr_for_predicted(thr_for_adjusted, optimal_threshold):
    """
    Returns a threshold that can be applied to the predicted probabilities so that
    predicted_proba > thr_for_predicted is equivalent to adjusted_proba > thr_for_adjusted

    :param thr_for_adjusted: threshold for class assignment in adjusted probabilities.
    :param optimal_threshold: optimal threshold for class assignment in predicted probabilities.
    :return: the equivalent threshold for class assignment in predicted probabilities
    """
    num = thr_for_adjusted * optimal_threshold
    den = thr_for_adjusted * optimal_threshold + (1.0 - thr_for_adjusted) * (1.0 - optimal_threshold)
    thr_for_predicted = num / den
    return thr_for_predicted


def get_event_probabilities(marks, probability, downsampling_factor=8, proba_prc=75):
    probability_upsampled = np.repeat(probability, downsampling_factor)
    # Retrieve segments of probabilities
    marks_segments = [probability_upsampled[m[0]:(m[1] + 1)] for m in marks]
    marks_proba = [np.percentile(m_seg, proba_prc) for m_seg in marks_segments]
    marks_proba = np.array(marks_proba)
    return marks_proba


def generate_ensemble_from_probabilities(dict_of_proba, reference_feeder_dataset, skip_setting_threshold=False):
    """
    dict_of_proba = {
        subject_id_1: list of probabilities to ensemble,
        subject_id_2: list of probabilities to ensemble,
        etc
    }
    """
    subject_ids = reference_feeder_dataset.get_ids()
    avg_dict = {}
    for subject_id in subject_ids:
        probabilities = np.stack(dict_of_proba[subject_id], axis=0).astype(np.float32).mean(axis=0).astype(np.float16)
        avg_dict[subject_id] = probabilities
    ensemble_prediction = PredictedDataset(
        dataset=reference_feeder_dataset,
        probabilities_dict=avg_dict,
        params=reference_feeder_dataset.params.copy(),
        skip_setting_threshold=skip_setting_threshold)
    return ensemble_prediction


def generate_ensemble_from_stamps(
        dict_of_stamps, reference_feeder_dataset, downsampling_factor=8, skip_setting_threshold=False):
    """
    dict_of_stamps = {
        subject_id_1: list of stamps to ensemble,
        subject_id_2: list of stamps to ensemble,
        etc
    }
    """
    subject_ids = reference_feeder_dataset.get_ids()
    dict_of_proba = {}
    for subject_id in subject_ids:
        stamps_list = dict_of_stamps[subject_id]
        subject_max_sample = np.max([
            (1 if single_stamp.size == 0 else single_stamp.max())
            for single_stamp in stamps_list])
        subject_max_sample = downsampling_factor * ((subject_max_sample // downsampling_factor) + 10)
        probabilities = [
            utils.stamp2seq(single_stamp, 0, subject_max_sample - 1).reshape(-1, downsampling_factor).mean(axis=1)
            for single_stamp in stamps_list]
        dict_of_proba[subject_id] = probabilities
    ensemble_prediction = generate_ensemble_from_probabilities(
        dict_of_proba, reference_feeder_dataset, skip_setting_threshold=skip_setting_threshold)
    return ensemble_prediction


def generate_ensemble_from_predicted_datasets(
        predicted_dataset_list,
        reference_feeder_dataset,
        use_probabilities=False,
        skip_setting_threshold=False
):
    subject_ids = reference_feeder_dataset.get_ids()
    dict_of_data = {}
    for subject_id in subject_ids:
        if use_probabilities:
            data_list = [
                pred.get_subject_probabilities(subject_id, return_adjusted=True)
                for pred in predicted_dataset_list]
        else:
            data_list = [
                pred.get_subject_stamps(subject_id)
                for pred in predicted_dataset_list]
        dict_of_data[subject_id] = data_list
    if use_probabilities:
        ensemble_prediction = generate_ensemble_from_probabilities(
            dict_of_data, reference_feeder_dataset, skip_setting_threshold=skip_setting_threshold)
    else:
        ensemble_prediction = generate_ensemble_from_stamps(
            dict_of_data, reference_feeder_dataset, skip_setting_threshold=skip_setting_threshold)
    return ensemble_prediction
