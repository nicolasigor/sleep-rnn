from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.data import utils, stamp_correction
from sleeprnn.detection import metrics
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(
    project_root, '../sleep-baselines/2017_lajnef_spinky'))

if __name__ == '__main__':

    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    fs = 128

    task_mode = constants.N2_RECORD
    id_try_list = np.arange(10)

    # Load expert annotations
    dataset = load_dataset(
        dataset_name, load_checkpoint=True, params={pkeys.FS: fs})

    display_setting = 'thr(33)'

    all_train_ids = dataset.train_ids
    page_size = dataset.page_size
    print('Page size:', page_size)
    print('All train ids', all_train_ids)
    gs_dict = {}
    n2_dict = {}
    for subject_id in all_train_ids:
        n2_dict[subject_id] = dataset.get_subject_pages(
            subject_id,
            pages_subset=task_mode)

    # Load predictions
    context_size = int(5 * fs)
    pred_folder = os.path.join(BASELINE_PATH, 'output_%s' % dataset_name)
    print('Loading predictions from %s' % pred_folder, flush=True)
    pred_files = os.listdir(pred_folder)
    pred_dict = {}
    detection_matrix_dict = {}
    for file in pred_files:
        subject_id = int(file.split('_')[1][1:])
        # only subjects for cross-validation
        if subject_id not in all_train_ids:
            continue

        setting = file.split('_')[2][:-4]

        if setting != display_setting:
            continue

        # Binary sequences
        filepath = os.path.join(pred_folder, file)
        pred_data = loadmat(filepath)
        pred_data = pred_data['detection_matrix']
        detection_matrix_dict[subject_id] = pred_data

        # Now we need to do different things if SS or KC
        if dataset_name == constants.MASS_SS_NAME:

            # Spindles
            start_sample = context_size
            end_sample = pred_data.shape[1] - context_size - 1
            pred_marks_list = []
            for page_idx in range(pred_data.shape[0]):
                page_sequence = pred_data[page_idx, :]
                page_marks = utils.seq2stamp(page_sequence)
                # we keep marks that are at least partially contained in page
                if page_marks.size > 0:
                    page_marks = utils.filter_stamps(page_marks, start_sample, end_sample)
                    if page_marks.size > 0:
                        # Before we append them, we need to provide the right
                        # sample
                        real_page = n2_dict[subject_id][page_idx]
                        offset_sample = int(real_page * page_size - context_size)
                        page_marks = page_marks + offset_sample
                        pred_marks_list.append(page_marks)
            if pred_marks_list:
                pred_marks = np.concatenate(pred_marks_list, axis=0)
                # Now we need to post-process
                pred_marks = stamp_correction.combine_close_stamps(
                    pred_marks, fs, min_separation=0.3)
                pred_marks = stamp_correction.filter_duration_stamps(
                    pred_marks, fs, min_duration=0.3, max_duration=3.0)
            else:
                pred_marks = np.array([]).reshape((0, 2))
            pred_dict[subject_id] = pred_marks

        elif dataset_name == constants.MASS_KC_NAME:
            # KC
            # We only keep what's inside the page
            pred_data_without_context = pred_data[:, context_size:-context_size]
            # Now we concatenate and then the extract stamps
            pred_marks = utils.seq2stamp_with_pages(
                pred_data_without_context, n2_dict[subject_id])
            #  Now we manually add 0.1 s before and 1.3 s after (paper)
            add_before = int(np.round(0.1 * fs))
            add_after = int(np.round(1.3 * fs))
            pred_marks[:, 0] = pred_marks[:, 0] - add_before
            pred_marks[:, 1] = pred_marks[:, 1] + add_after
            # By construction all marks are valid (inside N2 pages)
            # Save marks for evaluation
            pred_dict[subject_id] = pred_marks
        else:
            raise ValueError('Invalid dataset_name')

    # Start evaluation
    train_marks_list = dataset.get_subset_stamps(
        all_train_ids,
        which_expert=which_expert,
        pages_subset=task_mode)
    individual_performance = []
    iou_thr_list = np.linspace(0.05, 0.95, 19)
    for i, subject_id in enumerate(all_train_ids):
        # f1_bs = metrics.by_sample_confusion(
        #     train_marks_list[i], pred_dict[subject_id])[constants.F1_SCORE]
        f1_curve = metrics.metric_vs_iou(
            train_marks_list[i],
            pred_dict[subject_id],
            iou_thr_list,
            metric_name=constants.F1_SCORE,
            verbose=False
        )
        # print(subject_id, f1_bs)
        individual_performance.append(f1_curve)

    # Plot results
    plot_individual = False
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
    if plot_individual:
        for i, subject_id in enumerate(all_train_ids):
            ax.plot(iou_thr_list, individual_performance[i],
                    label=subject_id, linewidth=1)
        ax.legend(loc='upper right', fontsize=8)
    else:
        mean_curve = np.stack(individual_performance, axis=0).mean(axis=0)
        std_curve = np.stack(individual_performance, axis=0).std(axis=0)
        ax.plot(iou_thr_list, mean_curve, linewidth=1, color='b')
        ax.fill_between(iou_thr_list,
                        mean_curve - std_curve, mean_curve + std_curve,
                        facecolor='b', alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('IoU', fontsize=8)
    ax.set_ylabel('F1', fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(display_setting)
    plt.show()

    # # Check some annotations
    # window = 20
    # subject_id = 5
    #
    # which_mark = 50  # 26
    #
    # idx_is_prediction = True
    #
    # print('%d predictions for subject %d'
    #       % (pred_dict[subject_id].shape[0], subject_id))
    # signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
    # stamps = dataset.get_subject_stamps(
    #     subject_id,
    #     which_expert=which_expert,
    #     pages_subset=task_mode)
    #
    # if idx_is_prediction:
    #     single_mark = pred_dict[subject_id][which_mark, :]
    # else:
    #     single_mark = stamps[which_mark, :]
    #
    # start_sample_plot = int(single_mark.mean() - window * fs / 2)
    # end_sample_plot = int(start_sample_plot + window * fs)
    # segment_signal = signal[start_sample_plot:end_sample_plot]
    # plot_expert = utils.filter_stamps(stamps, start_sample_plot,
    #                                   end_sample_plot)
    # plot_pred = utils.filter_stamps(pred_dict[subject_id], start_sample_plot,
    #                                 end_sample_plot)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=80)
    # time_axis = np.arange(start_sample_plot, end_sample_plot) / fs
    # ax.plot(time_axis, segment_signal, linewidth=1)
    # for single_pred_mark in plot_pred:
    #     ax.fill_between(
    #         single_pred_mark / fs, -100, 100, facecolor='r', alpha=0.3
    #     )
    #     print(single_pred_mark)
    # for single_expert_mark in plot_expert:
    #     ax.plot(single_expert_mark / fs, [-100, -100],
    #             linewidth=5, color='b')
    # plt.show()

    # -------------
    # Show page
    # which_n2_page = 170
    # subject_id = 5
    #
    # signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
    # stamps = dataset.get_subject_stamps(
    #     subject_id,
    #     which_expert=which_expert,
    #     pages_subset=task_mode)
    # n2_pages = n2_dict[subject_id]#dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
    # real_page = n2_pages[which_n2_page]
    # start_sample_plot = int(real_page * page_size)
    # end_sample_plot = int((real_page + 1) * page_size)
    # segment_signal = signal[start_sample_plot:end_sample_plot]
    # plot_expert = utils.filter_stamps(stamps, start_sample_plot,
    #                                   end_sample_plot)
    # plot_pred = utils.filter_stamps(pred_dict[subject_id], start_sample_plot,
    #                                 end_sample_plot)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=80)
    # time_axis = np.arange(start_sample_plot, end_sample_plot) / fs
    # ax.plot(time_axis, segment_signal, linewidth=1)
    # for single_pred_mark in plot_pred:
    #     ax.fill_between(
    #         single_pred_mark / fs, -100, 100, facecolor='r', alpha=0.3
    #     )
    #     print(single_pred_mark)
    # for single_expert_mark in plot_expert:
    #     ax.plot(single_expert_mark / fs, [-100, -100],
    #             linewidth=5, color='b')
    #
    # ax.plot(time_axis, 120*detection_matrix_dict[subject_id][which_n2_page, context_size:-context_size])
    # plt.show()
