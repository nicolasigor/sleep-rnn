from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from joblib import delayed, Parallel

import numpy as np

from sleeprnn.detection import metrics


def get_optimal_threshold(
        feeder_dataset_list,
        predicted_dataset_list,
        res_thr=0.02,
        start_thr=0.3,
        end_thr=0.7,
        verbose=False
):

    # Check probability boundaries
    min_proba = start_thr  # 0
    max_proba = end_thr  # 1
    # for predicted_dataset in predicted_dataset_list:
    #     this_proba_list = predicted_dataset.get_probabilities()
    #     min_allowed = np.max([np.percentile(proba, 1) for proba in this_proba_list])
    #     max_allowed = np.min([np.percentile(proba, 99) for proba in this_proba_list])
    #     if min_allowed > min_proba:
    #         min_proba = min_allowed
    #     if max_allowed < max_proba:
    #         max_proba = max_allowed
    # min_proba = np.ceil(100 * min_proba) / 100
    # max_proba = np.floor(100 * max_proba) / 100

    # Change start_thr and end_thr accordingly
    start_thr = max(min_proba, start_thr)
    end_thr = min(max_proba, end_thr)
    if verbose:
        print('Start thr: %1.4f. End thr: %1.4f' % (start_thr, end_thr))
    n_thr = int(np.floor((end_thr - start_thr) / res_thr + 1))
    thr_list = np.array([start_thr + res_thr * i for i in range(n_thr)])
    thr_list = np.round(thr_list, 2)
    if verbose:
        print('%d thresholds to be evaluated between %1.4f and %1.4f'
              % (n_thr, thr_list[0], thr_list[-1]))

    events_list = []
    for feeder_dataset in feeder_dataset_list:
        # Prepare expert labels
        this_events = feeder_dataset.get_stamps()
        events_list = events_list + this_events

    predictions_at_thr_list = []
    for thr in thr_list:
        detections_list = []
        for predicted_dataset in predicted_dataset_list:
            # Prepare model predictions
            predicted_dataset.set_probability_threshold(thr)
            this_detections = predicted_dataset.get_stamps()
            detections_list = detections_list + this_detections
        predictions_at_thr_list.append(detections_list)

    af1_list = Parallel(n_jobs=-1)(
        delayed(metrics.average_metric_with_list)(
            events_list, single_prediction_list, verbose=False)
        for single_prediction_list in predictions_at_thr_list
    )

    # af1_list = [
    #     metrics.average_metric_with_list(
    #         events_list, single_prediction_list, verbose=False)
    #     for single_prediction_list in predictions_at_thr_list
    # ]


    # af1_list = []
    # for thr in thr_list:
    #     # events_list = []
    #     # detections_list = []
    #     # # for (feeder_dataset, predicted_dataset) in zip(
    #     # #         feeder_dataset_list, predicted_dataset_list):
    #     # for predicted_dataset in predicted_dataset_list:
    #     #     # Prepare expert labels
    #     #     # this_events = feeder_dataset.get_stamps()
    #     #     # Prepare model predictions
    #     #     predicted_dataset.set_probability_threshold(thr)
    #     #     this_detections = predicted_dataset.get_stamps()
    #     #     # events_list = events_list + this_events
    #     #     detections_list = detections_list + this_detections
    #     # Compute AF1
    #     af1_at_thr = metrics.average_metric_with_list(
    #         events_list, predictions_dict[thr], verbose=False)
    #     af1_list.append(af1_at_thr)

    max_idx = np.argmax(af1_list).item()
    best_thr = thr_list[max_idx]
    return best_thr
