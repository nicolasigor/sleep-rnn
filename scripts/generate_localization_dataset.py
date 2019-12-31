from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from pprint import pprint
import time

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import RefactorUnpickler, load_dataset
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.common import constants

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir

    prediction_folder = os.path.join(
        'predictions_mass_ss',
        '20190827_thesis_1_bsf_e1_n2_train_mass_ss',
        'v19')
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    id_try = 0
    optimal_thr_list = [0.54, 0.52, 0.64, 0.54]
    optimal_thr = optimal_thr_list[id_try]
    which_expert = 1

    segment_duration = 4

    # -----
    ckpt_path = os.path.join(RESULTS_PATH, prediction_folder, 'seed%d' % id_try)

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # Load data
    dataset = load_dataset(dataset_name)
    all_train_ids = dataset.train_ids
    test_ids = dataset.test_ids

    set_list = [
        constants.TRAIN_SUBSET,
        constants.VAL_SUBSET,
        constants.TEST_SUBSET]

    # Split to form validation set
    train_ids, val_ids = utils.split_ids_list_v2(all_train_ids, split_id=id_try)
    ids_dict = {
        constants.TRAIN_SUBSET: train_ids,
        constants.VAL_SUBSET: val_ids,
        constants.TEST_SUBSET: test_ids
    }

    fs = dataset.fs

    # Load predictions
    print('Loading predictions')
    pred_from_chosen_model = {}
    for set_name in set_list:
        this_dict = {}
        filename = os.path.join(
            ckpt_path,
            'prediction_%s_%s.pkl' % (task_mode, set_name))
        with open(filename, 'rb') as handle:
            pred_obj = RefactorUnpickler(handle).load()
            pred_obj.set_probability_threshold(optimal_thr)
            pred_from_chosen_model[set_name] = pred_obj
    print('Done\n')

    # # Compute performance
    # for set_name in set_list:
    #     data_inference = FeederDataset(
    #         dataset, ids_dict[set_name], task_mode,
    #         which_expert=which_expert)
    #     prediction_obj = pred_from_chosen_model[set_name]
    #
    #     this_events = data_inference.get_stamps()
    #     this_detections = prediction_obj.get_stamps()
    #     af1_at_thr = metrics.average_metric_with_list(
    #         this_events, this_detections, verbose=False)
    #     print('%s AF1: %1.4f' % (set_name, af1_at_thr))

    # Search for compatible events:
    duration_list = []
    distance_list = []
    space_list = []

    dataset_signal = {}
    dataset_marks = {}
    dataset_marks_binary = {}

    for set_name in set_list:
        print('Processing set %s' % set_name)
        data_inference = FeederDataset(
            dataset, ids_dict[set_name], task_mode,
            which_expert=which_expert)
        prediction_obj = pred_from_chosen_model[set_name]

        this_events = data_inference.get_stamps()
        this_detections = prediction_obj.get_stamps()

        set_iou_list = []

        set_segment_signal = {}
        set_segment_marks = {}
        set_segment_marks_binary = {}

        for i, subject_id in enumerate(ids_dict[set_name]):
            print('S%02d' % subject_id)
            single_events = this_events[i]
            single_detections = this_detections[i]
            iou_array, idx_array = metrics.matching(single_events, single_detections)
            valid_idx = np.where(idx_array != -1)[0]
            valid_iou = iou_array[valid_idx]

            set_iou_list.append(valid_iou)
            print('Selecting %d of %d (Recall %1.4f)'
                  % (len(valid_idx),
                     len(single_events),
                     len(valid_idx)/len(single_events)))

            subject_signal = dataset.get_subject_signal(
                subject_id, normalize_clip=False)
            subject_binary_mark = utils.stamp2seq(single_events, 0, len(subject_signal)-1)

            this_subject_segment_signal_list = []
            this_subject_segment_marks_list = []
            this_subject_segment_marks_binary_list = []

            for single_idx in valid_idx:
                this_duration = (single_events[single_idx, 1] - single_events[single_idx, 0]) / fs
                duration_list.append(this_duration)
                this_distances_list = []
                there_exists_distance = False
                if single_idx != 0:
                    # Compute left distance
                    left_distance = (single_events[single_idx, 0] - single_events[single_idx-1, 1]) / fs
                    if left_distance < 5:
                        this_distances_list.append(left_distance)
                        there_exists_distance = True
                if single_idx != len(single_events)-1:
                    # Compute right distance
                    right_distance = (single_events[single_idx + 1, 0] - single_events[single_idx, 1]) / fs
                    if right_distance < 5:
                        this_distances_list.append(right_distance)
                        there_exists_distance = True
                if there_exists_distance:
                    distance_list.append(np.min(this_distances_list))

                if len(this_distances_list) == 2:
                    space = np.sum(this_distances_list) + this_duration
                    space_list.append(space)

                # Build crop
                min_distance_to_border = fs * 0.7
                this_size = single_events[single_idx, 1] - single_events[single_idx, 0]
                empty_space = segment_duration * fs - this_size
                if single_idx == 0:
                    right_distance = single_events[single_idx + 1, 0] - single_events[single_idx, 1] - 1
                    left_distance = 1000
                elif single_idx == len(single_events)-1:
                    left_distance = single_events[single_idx, 0] - single_events[single_idx - 1, 1] - 1
                    right_distance = 1000
                else:
                    right_distance = single_events[single_idx + 1, 0] - \
                                     single_events[single_idx, 1] - 1
                    left_distance = single_events[single_idx, 0] - \
                                    single_events[single_idx - 1, 1] - 1
                if right_distance < min_distance_to_border or left_distance < min_distance_to_border or right_distance + left_distance < empty_space:
                    print('Dropping mark')
                else:
                    # Let's crop
                    left_space = np.random.uniform(
                        low=max(min_distance_to_border, empty_space - right_distance),
                        high=min(empty_space - min_distance_to_border, left_distance))
                    start_sample = int(single_events[single_idx, 0] - left_space)
                    end_sample = int(start_sample + segment_duration * fs)

                    chosen_signal = subject_signal[start_sample:end_sample]
                    chosen_event = single_events[single_idx, :] - start_sample
                    chosen_binary = subject_binary_mark[start_sample:end_sample]

                    this_subject_segment_signal_list.append(chosen_signal)
                    this_subject_segment_marks_list.append(chosen_event)
                    this_subject_segment_marks_binary_list.append(chosen_binary)

            # sum_binary_subject = np.stack(this_subject_segment_marks_binary_list, axis=0).sum(axis=0)
            # plt.title('S%02d' % subject_id)
            # plt.bar(np.arange(segment_duration * fs), sum_binary_subject)
            # plt.show()

            this_subject_segment_signal_list = np.stack(this_subject_segment_signal_list, axis=0)
            this_subject_segment_marks_list = np.stack(this_subject_segment_marks_list, axis=0)
            this_subject_segment_marks_binary_list = np.stack(this_subject_segment_marks_binary_list, axis=0)

            set_segment_signal[subject_id] = this_subject_segment_signal_list
            set_segment_marks[subject_id] = this_subject_segment_marks_list
            set_segment_marks_binary[subject_id] = this_subject_segment_marks_binary_list

        dataset_signal[set_name] = set_segment_signal
        dataset_marks[set_name] = set_segment_marks
        dataset_marks_binary[set_name] = set_segment_marks_binary

        set_mean_iou = np.mean(np.concatenate(set_iou_list))
        print('Mean IoU %1.4f' % set_mean_iou)
        print('')

    # plt.title('Duration of events [s]')
    # plt.hist(duration_list, bins=20)
    # plt.show()
    #
    # plt.title('Distance between events [s]')
    # plt.hist(distance_list, bins=20)
    # plt.show()
    #
    # plt.title('Croppable space [s]')
    # plt.hist(space_list, bins=20)
    # plt.show()
    #
    # print('Max duration:', np.max(duration_list))
    # print('Number of ')
    print(dataset_signal.keys())
    for set_name in set_list:
        print(set_name)
        print(dataset_signal[set_name].keys())
        subjects_in_set = list(dataset_signal[set_name].keys())
        print('visiting', subjects_in_set[0])
        print(dataset_signal[set_name][subjects_in_set[0]].shape)
        print(dataset_marks[set_name][subjects_in_set[0]].shape)
        print(dataset_marks_binary[set_name][subjects_in_set[0]].shape)

    # Save data
    for set_name in set_list:

        concat_signals = []
        concat_marks = []
        concat_marks_binary = []

        for subject_id in dataset_signal[set_name].keys():
            concat_signals.append(dataset_signal[set_name][subject_id])
            concat_marks.append(dataset_marks[set_name][subject_id])
            concat_marks_binary.append(dataset_marks_binary[set_name][subject_id])

        concat_signals = np.concatenate(concat_signals, axis=0).astype(np.float32)
        concat_marks = np.concatenate(concat_marks, axis=0).astype(np.int32)
        concat_marks_binary = np.concatenate(concat_marks_binary, axis=0).astype(np.int32)
        print(set_name)
        print(concat_signals.shape)
        print(concat_marks.shape)
        print(concat_marks_binary.shape)

        np.save('%s_signals.npy' % set_name, concat_signals)
        np.save('%s_marks.npy' % set_name, concat_marks)
        np.save('%s_marks_binary.npy' % set_name, concat_marks_binary)
