"""eval_ops.py: Module for general evaluation metrics operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleep.data_ops import inter2seq, seq2inter


def by_sample_confusion(events, detections, input_is_binary=False):
    """Returns a dictionary with by-sample metrics.
    If input_is_binary is true, the inputs are assumed to be binary sequences. If False, is assumed to be sample-stamps
    in ascending order.
    """
    # We need binary sequences here, so let's transform if that's not the case
    if not input_is_binary:
        last_sample = max(events[-1, 1], detections[-1, 1])
        events = inter2seq(events, 0, last_sample)
        detections = inter2seq(detections, 0, last_sample)
    tp = np.sum((events == 1) & (detections == 1))
    fp = np.sum((events == 0) & (detections == 1))
    fn = np.sum((events == 1) & (detections == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    by_sample_metrics = {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1_score': f1_score
    }
    return by_sample_metrics


def by_sample_iou(events, detections, input_is_binary=False):
    """Returns the IoU considering the entire eeg as a single segmentation problem.
    If input_is_binary is true, the inputs are assumed to be binary sequences. If False, is assumed to be sample-stamps
    in ascending order.
    """
    # We need binary sequences here, so let's transform if that's not the case
    if not input_is_binary:
        last_sample = max(events[-1, 1], detections[-1, 1])
        events = inter2seq(events, 0, last_sample)
        detections = inter2seq(detections, 0, last_sample)
    intersection = np.sum((events == 1) & (detections == 1))
    sum_areas = np.sum(events) + np.sum(detections)
    union = sum_areas - intersection
    iou = intersection / union
    return iou


def by_event_confusion(events, detections, iou_thr=0.3, iou_array=None):
    """Returns a dictionary with by-events metrics.
    events and detections are assumed to be sample-stamps, and to be in ascending order.
    iou_array can be provided if it is already computed. If this is the case, events and detections are ignored.
    """
    if iou_array is None:
        iou_array, _ = matching(events, detections)
    mean_all_iou = np.mean(iou_array)
    # First, remove the zero iou_array entries
    iou_array = iou_array[iou_array > 0]
    mean_nonzero_iou = np.mean(iou_array)
    # Now, give credit only for iou >= iou_thr
    tp = np.sum((iou_array >= iou_thr).astype(int))
    n_detections = detections.shape[0]
    n_events = events.shape[0]
    precision = tp / n_detections
    recall = tp / n_events
    f1_score = 2 * precision * recall / (precision + recall)
    by_event_metrics = {
        'tp': tp, 'n_detections': n_detections, 'n_events': n_events,
        'precision': precision, 'recall': recall, 'f1_score': f1_score,
        'mean_all_iou': mean_all_iou, 'mean_nonzero_iou': mean_nonzero_iou
    }
    return by_event_metrics


def matching(events, detections):
    """Returns the IoU associated with each event. Events that has no detections have IoU zero.
    events and detections are assumed to be sample-stamps, and to be in ascending order."""
    # Matrix of overlap, rows are events, columns are detections
    n_det = detections.shape[0]
    n_gs = events.shape[0]
    overlaps = np.zeros((n_gs, n_det))
    for i in range(n_gs):
        for j in range(n_det):
            inter_samples = np.arange(max(events[i, 0], detections[j, 0]), min(events[i, 1], detections[j, 1]) + 1)
            if inter_samples.size > 0:
                intersection = inter_samples.size
                union_samples = np.arange(min(events[i, 0], detections[j, 0]), max(events[i, 1], detections[j, 1]) + 1)
                union = union_samples.size
                overlaps[i, j] = intersection / union
    # Greedy matching
    iou_array = []  # Array for IoU for every true event (gs)
    idx_array = []  # Array for the index associated with the true event. If no detection is found, this value is -1
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
