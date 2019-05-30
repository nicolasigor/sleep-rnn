from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba import jit, prange

from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection.predicted_dataset import PredictedDataset


def get_optimal_threshold(
        feeder_dataset_list,
        predicted_dataset_list,
        res_thr=0.02,
        start_thr=0.3,
        end_thr=0.7,
        verbose=False
):
    n_thr = int(np.round((end_thr - start_thr) / res_thr + 1))
    thr_list = np.array([start_thr + res_thr * i for i in range(n_thr)])
    thr_list = np.round(thr_list, 2)
    if verbose:
        print('%d thresholds to be evaluated between %1.4f and %1.4f'
              % (n_thr, thr_list[0], thr_list[-1]))

    af1_list = []
    for thr in thr_list:
        events_list = []
        detections_list = []
        for (feeder_dataset, predicted_dataset) in zip(
                feeder_dataset_list, predicted_dataset_list):
            # Prepare expert labels
            this_events = feeder_dataset.get_stamps()
            # Prepare model predictions
            predicted_dataset.set_probability_threshold(thr)
            this_detections = predicted_dataset.get_stamps()
            events_list = events_list + this_events
            detections_list = detections_list + this_detections
        # Compute AF1
        af1_at_thr = metrics.average_metric_with_list(
            events_list, detections_list, verbose=False)
        af1_list.append(af1_at_thr)
    max_idx = np.argmax(af1_list).item()
    best_thr = thr_list[max_idx]
    return best_thr
