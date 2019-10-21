from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.data import utils
from sleeprnn.detection import metrics
from sleeprnn.common import constants, pkeys

BASELINE_PATH = os.path.abspath(os.path.join(project_root, '../sleep-baselines/2019_lacourse'))

if __name__ == '__main__':

    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    fs = 128

    task_mode = constants.N2_RECORD

    display_setting = 'absSigPow(1.25)_relSigPow(1.6)_sigCov(1.3)_sigCorr(0.69)'

    # Load expert annotations
    dataset_params = {pkeys.FS: fs}
    dataset = load_dataset(
        dataset_name,
        params=dataset_params, load_checkpoint=False)
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
    pred_folder = os.path.join(BASELINE_PATH, 'output_%s' % dataset_name)
    pred_files = os.listdir(pred_folder)
    pred_dict = {}
    for file in pred_files:
        subject_id = int(file.split('_')[3][1:])
        setting = '_'.join(file.split('_')[4:])[:-4]
        if setting == display_setting:
            # sample marks
            filepath = os.path.join(pred_folder, file)
            pred_data = pd.read_csv(filepath, sep='\t')
            # We substract 1 to translate from matlab to numpy indexing system
            start_samples = pred_data.start_sample.values - 1
            end_samples = pred_data.end_sample.values - 1
            pred_marks = np.stack([start_samples, end_samples], axis=1)
            # Valid subset of marks
            pred_marks_n2 = utils.extract_pages_for_stamps(
                pred_marks, n2_dict[subject_id], page_size)
            # Save marks for evaluation
            pred_dict[subject_id] = pred_marks_n2

    # # Start evaluation
    print('Starting evaluation of setting %s' % display_setting, flush=True)
    train_marks_list = dataset.get_subset_stamps(
        all_train_ids,
        which_expert=which_expert,
        pages_subset=task_mode)
    individual_performance = []
    iou_thr_list = np.linspace(0.05, 0.95, 19)
    for i, subject_id in enumerate(all_train_ids):
        f1_curve = metrics.metric_vs_iou(
            train_marks_list[i],
            pred_dict[subject_id],
            iou_thr_list,
            metric_name=constants.F1_SCORE,
            verbose=False
        )
        individual_performance.append(f1_curve)

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
    for i, subject_id in enumerate(all_train_ids):
        ax.plot(iou_thr_list, individual_performance[i], label=subject_id, linewidth=1)
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('IoU', fontsize=8)
    ax.set_ylabel('F1', fontsize=8)
    plt.show()

    # Check some annotations
    window = 20
    subject_id = 1

    which_mark = 102  #26

    idx_is_prediction = False

    print('%d predictions for subject %d'
          % (pred_dict[subject_id].shape[0], subject_id))
    signal = dataset.get_subject_signal(subject_id, normalize_clip=False)
    stamps = dataset.get_subject_stamps(
        subject_id,
        which_expert=which_expert,
        pages_subset=task_mode)

    if idx_is_prediction:
        single_mark = pred_dict[subject_id][which_mark, :]
    else:
        single_mark = stamps[which_mark, :]

    start_sample_plot = int(single_mark.mean() - window * fs / 2)
    end_sample_plot = int(start_sample_plot + window * fs)
    segment_signal = signal[start_sample_plot:end_sample_plot]
    plot_expert = utils.filter_stamps(stamps, start_sample_plot, end_sample_plot)
    plot_pred = utils.filter_stamps(pred_dict[subject_id], start_sample_plot, end_sample_plot)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=80)
    time_axis = np.arange(start_sample_plot, end_sample_plot) / fs
    ax.plot(time_axis, segment_signal, linewidth=1)
    for single_pred_mark in plot_pred:
        ax.fill_between(
            single_pred_mark / fs, -100, 100, facecolor='r', alpha=0.3
        )
        print(single_pred_mark)
    for single_expert_mark in plot_expert:
        ax.plot(single_expert_mark / fs, [-100, -100],
                linewidth=5, color='b')
    plt.show()
