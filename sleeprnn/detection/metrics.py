"""metrics.py: Module for general evaluation metrics operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleeprnn.common import constants
from sleeprnn.data.utils import stamp2seq


def by_sample_confusion(events, detections, input_is_binary=False):
    """Returns a dictionary with by-sample metrics.
    If input_is_binary is true, the inputs are assumed to be binary sequences.
    If False, is assumed to be sample-stamps
    in ascending order.
    """
    # We need binary sequences here, so let's transform if that's not the case
    if not input_is_binary:
        last_sample = max(events[-1, 1], detections[-1, 1])
        events = stamp2seq(events, 0, last_sample)
        detections = stamp2seq(detections, 0, last_sample)
    tp = np.sum((events == 1) & (detections == 1))
    fp = np.sum((events == 0) & (detections == 1))
    fn = np.sum((events == 1) & (detections == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    by_sample_metrics = {
        constants.TP: tp,
        constants.FP: fp,
        constants.FN: fn,
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1_SCORE: f1_score
    }
    return by_sample_metrics


def by_sample_iou(events, detections, input_is_binary=False):
    """Returns the IoU considering the entire eeg as a single segmentation
    problem.
    If input_is_binary is true, the inputs are assumed to be binary sequences.
    If False, is assumed to be sample-stamps
    in ascending order.
    """
    # We need binary sequences here, so let's transform if that's not the case
    if not input_is_binary:
        last_sample = max(events[-1, 1], detections[-1, 1])
        events = stamp2seq(events, 0, last_sample)
        detections = stamp2seq(detections, 0, last_sample)
    intersection = np.sum((events == 1) & (detections == 1))
    sum_areas = np.sum(events) + np.sum(detections)
    union = sum_areas - intersection
    iou = intersection / union
    return iou


def by_event_confusion(events, detections, iou_thr=0.3, iou_array=None):
    """Returns a dictionary with by-events metrics.
    events and detections are assumed to be sample-stamps, and to be in
    ascending order.
    iou_array can be provided if it is already computed. If this is the case,
    events and detections are ignored.
    """
    if iou_array is None:
        iou_array, _ = matching(events, detections)
    n_detections = detections.shape[0]
    n_events = events.shape[0]
    mean_all_iou = np.mean(iou_array)
    # First, remove the zero iou_array entries
    iou_array = iou_array[iou_array > 0]
    mean_nonzero_iou = np.mean(iou_array)
    # Now, give credit only for iou >= iou_thr
    tp = np.sum((iou_array >= iou_thr).astype(int))
    fp = n_detections - tp
    fn = n_events - tp
    precision = tp / n_detections
    recall = tp / n_events

    # f1-score is 2 * precision * recall / (precision + recall),
    # but considering the case tp=0, a more stable formula is:
    f1_score = 2 * tp / (n_detections + n_events)

    by_event_metrics = {
        constants.TP: tp,
        constants.FP: fp,
        constants.FN: fn,
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1_SCORE: f1_score,
        constants.MEAN_ALL_IOU: mean_all_iou,
        constants.MEAN_NONZERO_IOU: mean_nonzero_iou
    }
    return by_event_metrics


def matching(events, detections):
    """Returns the IoU associated with each event. Events that has no detections
    have IoU zero. events and detections are assumed to be sample-stamps, and to
    be in ascending order."""
    # Matrix of overlap, rows are events, columns are detections
    n_det = detections.shape[0]
    n_gs = events.shape[0]
    overlaps = np.zeros((n_gs, n_det))
    for i in range(n_gs):
        candidates = np.where(
            (detections[:, 0] <= events[i, 1])
            & (detections[:, 1] >= events[i, 0]))[0]
        for j in candidates:
            intersection = min(
                events[i, 1], detections[j, 1]
            ) - max(
                events[i, 0], detections[j, 0]
            ) + 1
            if intersection > 0:
                union = max(
                    events[i, 1], detections[j, 1]
                ) - min(
                    events[i, 0], detections[j, 0]
                ) + 1
                overlaps[i, j] = intersection / union
    # Greedy matching
    iou_array = []  # Array for IoU for every true event (gs)
    idx_array = []  # Array for the index associated with the true event.
    # If no detection is found, this value is -1
    for i in range(n_gs):
        if np.sum(overlaps[i, :]) > 0:
            # Find max overlap
            max_j = np.argmax(overlaps[i, :])
            iou_array.append(overlaps[i, max_j])
            idx_array.append(max_j)
            # Remove this detection for further search
            overlaps[i, max_j] = 0
        else:
            iou_array.append(0)
            idx_array.append(-1)
    iou_array = np.array(iou_array)
    idx_array = np.array(idx_array)
    return iou_array, idx_array


def metric_vs_iou(
        events,
        detections,
        iou_thr_list,
        metric_name=constants.F1_SCORE,
        verbose=False
):
    metric_list = []
    if verbose:
        print('Matching events... ', end='', flush=True)
    this_iou_data, _ = matching(events, detections)
    if verbose:
        print('Done', flush=True)
    for iou_thr in iou_thr_list:
        if verbose:
            print(
                'Processing IoU threshold %1.4f... ' % iou_thr,
                end='', flush=True)
        this_stat = by_event_confusion(
            events, detections,
            iou_thr=iou_thr, iou_array=this_iou_data)
        metric = this_stat[metric_name]
        metric_list.append(metric)
        if verbose:
            print('%s obtained: %1.4f' % (metric_name, metric), flush=True)
    if verbose:
        print('Done')
    metric_list = np.array(metric_list)
    return metric_list


def metric_vs_iou_with_list(
        events_list,
        detections_list,
        iou_thr_list,
        metric_name=constants.F1_SCORE,
        verbose=False
):
    all_metric_list = []
    for events, detections in zip(events_list, detections_list):
        metric_list = metric_vs_iou(
            events, detections, iou_thr_list,
            metric_name=metric_name, verbose=verbose)
        all_metric_list.append(metric_list)
    all_metric_curve = np.stack(all_metric_list, axis=1).mean(axis=1)
    return all_metric_curve


def average_metric(
        events,
        detections,
        metric_name=constants.F1_SCORE,
        verbose=False
):
    """Average F1 over several IoU values.

    The average F1 performance is
    computed as the area under the F1 vs IoU curve.
    """
    # Go through several IoU values
    first_iou = 0
    last_iou = 1
    res_iou = 0.01
    n_points = int(np.round((last_iou - first_iou) / res_iou))
    full_iou_list = np.arange(n_points + 1) * res_iou + first_iou
    if verbose:
        print('Using %d IoU thresholds from %1.1f to %1.1f'
              % (n_points + 1, first_iou, last_iou))
        print('Computing %s values' % metric_name, flush=True)

    metric_list = metric_vs_iou(
        events, detections, full_iou_list,
        metric_name=metric_name, verbose=verbose)

    # To compute the area under the curve, we'll use trapezoidal aproximation
    # So we need to divide by two the extremes
    metric_list[0] = metric_list[0] / 2
    metric_list[-1] = metric_list[-1] / 2
    # And now we compute the AUC
    avg_metric = np.sum(metric_list * res_iou)
    if verbose:
        print('Done')
    return avg_metric


def average_metric_with_list(
        events_list,
        detections_list,
        metric_name=constants.F1_SCORE,
        verbose=False
):
    all_avg_list = []
    for events, detections in zip(events_list, detections_list):
        avg_metric = average_metric(
            events, detections,
            metric_name=metric_name, verbose=verbose)
        all_avg_list.append(avg_metric)
    all_avg = np.mean(all_avg_list)
    return all_avg
