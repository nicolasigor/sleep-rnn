from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils

MODA_PATH = '../resources/datasets/moda'


def get_subjects():
    p1_info = pd.read_csv(os.path.join(MODA_PATH, '6_segListSrcDataLoc_p1.txt'), delimiter='\t')
    p1_subjects = np.unique(p1_info.subjectID.values)
    p2_info = pd.read_csv(os.path.join(MODA_PATH, '7_segListSrcDataLoc_p2.txt'), delimiter='\t')
    p2_subjects = np.unique(p2_info.subjectID.values)
    phase_dict = {}
    for subject_id in p1_subjects:
        phase_dict[subject_id] = 1
    for subject_id in p2_subjects:
        phase_dict[subject_id] = 2
    subject_ids = list(phase_dict.keys())
    subject_ids.sort()
    return subject_ids, phase_dict


def get_data(subject_id):
    data = np.load(os.path.join(MODA_PATH, 'signals_npz/moda_%s.npz' % subject_id))
    fs = data['sampling_rate'].item()
    channel = data['channel'].item()
    signal = data['signal']
    return signal, fs, channel


def get_annotations(subject_id):
    annot = pd.read_csv(os.path.join(MODA_PATH, 'MODA_annotFiles/%s_MODA_GS.txt' % subject_id), delimiter='\t')
    segments_info = annot[annot.eventName == 'segmentViewed']
    segments_start = np.sort(segments_info.startSec.values)
    spindles_info = annot[annot.eventName == 'spindle']
    spindles_start = spindles_info.startSec.values
    spindles_end = spindles_info.durationSec.values + spindles_start
    return segments_start, spindles_start, spindles_end


def get_segment(x, fs, start_time, segment_duration=115, border_duration=30, border_to_filter_duration=10):
    # Extract a single segment of EEG
    total_border = border_duration * fs + border_to_filter_duration * fs
    segment_size = 2 * total_border + segment_duration * fs
    start_sample = int(start_time * fs - total_border)
    end_sample = int(start_sample + segment_size)
    segment = x[start_sample:end_sample].copy()
    return segment


def filter_segment(x, fs, lowcut=0.3, highcut=30, filter_order=10, border_to_filter_duration=10):
    sos = butter(filter_order, lowcut, btype='highpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x)
    sos = butter(filter_order, highcut, btype='lowpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x)
    border_size = int(fs * border_to_filter_duration)
    return x[border_size:-border_size]


def get_label(binaries, fs, start_time, segment_duration=115, border_duration=30):
    border_size = int(fs * border_duration)
    segment_size = 2 * border_size + segment_duration * fs
    start_sample = int(start_time * fs - border_size)
    end_sample = int(start_sample + segment_size)
    labels = binaries[start_sample:end_sample].copy()
    labels[:border_size] = -1
    labels[-border_size:] = -1

    return labels


if __name__ == "__main__":
    save_dir = "../resources/datasets/moda/segments"
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("Files will be saved at %s" % save_dir)

    metadata_l = []
    segments_signal_l = []
    segments_labels_l = []
    segments_subjects_l = []
    segments_phases_l = []

    subject_ids, phase_dict = get_subjects()
    for subject_id in subject_ids:
        print(subject_id, flush=True)
        signal, fs, channel = get_data(subject_id)
        segments_start, spindles_start, spindles_end = get_annotations(subject_id)
        stamps_time = np.stack([spindles_start, spindles_end], axis=1)
        stamps = (stamps_time * fs).astype(np.int32)
        binary_labels = utils.stamp2seq(stamps, 0, signal.size-1)
        for single_start in segments_start:
            segment_signal_prefilter = get_segment(signal, fs, single_start)
            segment_signal = filter_segment(segment_signal_prefilter, fs).astype(np.float32)
            segment_label = get_label(binary_labels, fs, single_start).astype(np.int8)
            # Append data
            segments_signal_l.append(segment_signal)
            segments_labels_l.append(segment_label)
            segments_subjects_l.append(subject_id)
            segments_phases_l.append(phase_dict[subject_id])
            metadata_l.append({
                'subject_id': subject_id,
                'phase': phase_dict[subject_id],
                'channel': channel,
                'fs': fs,
                'start_seconds': single_start,
                'segment_seconds': 115,
                'border_seconds': 30
            })

    # Format data
    segments_signal = np.stack(segments_signal_l, axis=0)
    segments_labels = np.stack(segments_labels_l, axis=0)
    segments_subjects = np.stack(segments_subjects_l, axis=0)
    segments_phases = np.stack(segments_phases_l, axis=0).astype(np.int8)
    metadata_table = pd.DataFrame(metadata_l)

    # Save data
    np.savez(
        os.path.join(save_dir, "moda_preprocessed_segments.npz"),
        signals=segments_signal, labels=segments_labels, subjects=segments_subjects, phases=segments_phases)
    metadata_table.to_csv(os.path.join(save_dir, "metadata.csv"), sep='\t')
