from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib import gridspec
import matplotlib.image as mpimg

project_root = '..'
sys.path.append(project_root)

# from sleeprnn.data.inta_ss import IntaSS, NAMES
from sleeprnn.data import utils, stamp_correction
from sleeprnn.detection import metrics
from sleeprnn.helpers import reader, misc, plotter
from sleeprnn.common import constants, pkeys, viz


NAMES = [
    'ADGU101504',
    'ALUR012904',
    'BECA011405',
    'BRCA062405',
    'BRLO041102',
    'BTOL083105',
    'BTOL090105',
    'CAPO092605',
    'CRCA020205',
    'ESCI031905',
    'TAGO061203']

# Order: (from worst to best)
# 11 TAGO [x] | 269 conflict pages
# 08 CAPO [x] | 82
# 02 ALUR [x] | 314
# 06 BTOL08 | 9
# 04 BRCA | 479
# 09 CRCA | 308
# 10 ESCI | 69
# 05 BRLO | 156
# 07 BTOL09 | 22
# 03 BECA | 3
# 01 ADGU | 232

# Load subject
# subject_id = 2
save_selected_pages = True
# Sleep states dictionary for INTA:
# 1:SQ4   2:SQ3   3:SQ2   4:SQ1   5:REM   6:WA
inta_stages_names = {1: 'SQ4', 2: 'SQ3', 3: 'SQ2', 4: 'SQ1', 5: 'REM', 6: 'WA'}
stages_valid = [3]
subject_ids = [3]

for subject_id in subject_ids:
    fs = 200
    marked_channel = 'F4-C4'
    dataset_dir = os.path.abspath(os.path.join('..', 'resources/datasets/inta'))
    page_duration = 20
    page_size = page_duration * fs

    print('Loading S%02d' % subject_id)
    path_stamps = os.path.join(dataset_dir, 'label/spindle/original/', 'SS_%s.txt' % NAMES[subject_id - 1])
    path_signals = os.path.join(dataset_dir, 'register', '%s.rec' % NAMES[subject_id - 1])
    signal_dict = reader.read_signals_from_edf(path_signals)
    signal_names = list(signal_dict.keys())
    to_show_names = misc.get_inta_eeg_names(signal_names) + misc.get_inta_eog_emg_names(signal_names)
    for single_name in misc.get_inta_eeg_names(signal_names):
        this_signal = signal_dict[single_name]
        print('Filtering %s channel' % single_name)
        this_signal = utils.broad_filter(this_signal, fs)
        signal_dict[single_name] = this_signal
    raw_stamps_1, raw_stamps_2 = reader.load_raw_inta_stamps(path_stamps, path_signals, min_samples=20, chn_idx=0)
    durations_1 = (raw_stamps_1[:, 1] - raw_stamps_1[:, 0]) / fs
    durations_2 = (raw_stamps_2[:, 1] - raw_stamps_2[:, 0]) / fs
    print('V1', raw_stamps_1.shape, 'Min dur [s]', durations_1.min(), 'Max dur [s]', durations_1.max())
    print('V2', raw_stamps_2.shape, 'Min dur [s]', durations_2.min(), 'Max dur [s]', durations_2.max())
    overlap_m = utils.get_overlap_matrix(raw_stamps_1, raw_stamps_1)
    groups_overlap_1 = utils.overlapping_groups(overlap_m)
    overlap_m = utils.get_overlap_matrix(raw_stamps_2, raw_stamps_2)
    groups_overlap_2 = utils.overlapping_groups(overlap_m)
    n_overlaps_1 = [len(single_group) for single_group in groups_overlap_1]
    values_1, counts_1 = np.unique(n_overlaps_1, return_counts=True)
    print('\nSize of overlapping groups for Valid 1')
    for value, count in zip(values_1, counts_1):
        print('%d marks: %d times' % (value, count))
    n_overlaps_2 = [len(single_group) for single_group in groups_overlap_2]
    values_2, counts_2 = np.unique(n_overlaps_2, return_counts=True)
    print('\nSize of overlapping groups for Valid 2')
    for value, count in zip(values_2, counts_2):
        print('%d marks: %d times' % (value, count))
    max_overlaps = np.max([values_1.max(), values_2.max()]) - 1
    this_pages = np.arange(1, signal_dict[marked_channel].size//page_size - 1)
    print('This pages', this_pages.shape)

    # Select marks without doubt
    groups_in_doubt_v1_list = []
    groups_in_doubt_v2_list = []

    iou_to_accept = 0.8
    marks_without_doubt = []
    overlap_between_1_and_2 = utils.get_overlap_matrix(raw_stamps_1, raw_stamps_2)

    for single_group in groups_overlap_2:
        if len(single_group) == 1:
            marks_without_doubt.append(raw_stamps_2[single_group[0], :])
        elif len(single_group) == 2:
            # check if IOU between marks is close 1, if close, then just choose newer (second one)
            option1_mark = raw_stamps_2[single_group[0], :]
            option2_mark = raw_stamps_2[single_group[1], :]
            iou_between_marks = metrics.get_iou(option1_mark, option2_mark)
            if iou_between_marks >= iou_to_accept:
                marks_without_doubt.append(option2_mark)
            else:
                groups_in_doubt_v2_list.append(single_group)
        else:
            groups_in_doubt_v2_list.append(single_group)

    for single_group in groups_overlap_1:
        is_in_doubt = False
        # Check if entire group is overlapping
        all_are_overlapping_2 = np.all(overlap_between_1_and_2[single_group, :].sum(axis=1))
        if not all_are_overlapping_2:
            # Consider the mark
            if len(single_group) == 1:
                # Since has size 1 and is no overlapping 2, accept it
                marks_without_doubt.append(raw_stamps_1[single_group[0], :])
            elif len(single_group) == 2:
                # check if IOU between marks is close 1, if close, then just choose newer (second one) since there is no intersection
                option1_mark = raw_stamps_1[single_group[0], :]
                option2_mark = raw_stamps_1[single_group[1], :]
                iou_between_marks = metrics.get_iou(option1_mark, option2_mark)
                if iou_between_marks >= iou_to_accept:
                    marks_without_doubt.append(raw_stamps_1[single_group[1], :])
                else:
                    is_in_doubt = True
            else:
                is_in_doubt = True
        if is_in_doubt:
            groups_in_doubt_v1_list.append(single_group)

    marks_without_doubt = np.stack(marks_without_doubt, axis=0)
    marks_without_doubt = np.sort(marks_without_doubt, axis=0)
    print('Marks automatically added:', marks_without_doubt.shape)
    print('Remaining conflicts:')
    print('    V1: %d' % len(groups_in_doubt_v1_list))
    print('    V2: %d' % len(groups_in_doubt_v2_list))

    show_complete_conflict_detail = False

    conflict_pages = []

    if show_complete_conflict_detail:
        print('Conflict detail')
    for single_group in groups_in_doubt_v1_list:
        group_stamps = raw_stamps_1[single_group, :]
        min_sample = group_stamps.min()
        max_sample = group_stamps.max()
        center_group = (min_sample + max_sample) / 2
        integer_page = int(center_group / page_size)
        decimal_part = np.round(2 * (center_group % page_size) / page_size) / 2 - 0.5
        page_location = integer_page + decimal_part
        conflict_pages.append(page_location)
        if show_complete_conflict_detail:
            print('V1 - Group of size %d at page %1.1f' % (group_stamps.shape[0], page_location ))

    for single_group in groups_in_doubt_v2_list:
        group_stamps = raw_stamps_2[single_group, :]
        min_sample = group_stamps.min()
        max_sample = group_stamps.max()
        center_group = (min_sample + max_sample) / 2
        integer_page = int(center_group / page_size)
        decimal_part = np.round(2 * (center_group % page_size) / page_size) / 2 - 0.5
        page_location = integer_page + decimal_part
        conflict_pages.append(page_location)
        if show_complete_conflict_detail:
            print('V2 - Group of size %d at page %1.1f' % (group_stamps.shape[0], page_location ))
    conflict_pages = np.unique(conflict_pages)

    print('')
    print('Number of pages with conflict %d' % conflict_pages.size)

    this_final_marks = np.array([])


    def plot_page_conflict(page_chosen, ax, show_final=False):
        # conflict id starts from 1.
        signal_uv_to_display = 20
        microvolt_per_second = 200  # Aspect ratio
        page_start = page_chosen * page_size
        page_end = page_start + page_size
        segment_stamps = utils.filter_stamps(marks_without_doubt, page_start, page_end)
        segment_stamps_valid_1 = utils.filter_stamps(raw_stamps_1, page_start, page_end)
        segment_stamps_valid_2 = utils.filter_stamps(raw_stamps_2, page_start, page_end)
        segment_stamps_final = utils.filter_stamps(this_final_marks, page_start, page_end) if show_final else []
        time_axis = np.arange(page_start, page_end) / fs
        x_ticks = np.arange(time_axis[0], time_axis[-1]+1, 1)
        dy_valid = 40
        shown_valid = False
        valid_label = 'Candidate mark'
        # Show valid 1
        valid_start = -100
        shown_groups_1 = []
        for j, this_stamp in enumerate(segment_stamps_valid_1):
            idx_stamp = np.where([np.all(this_stamp == single_stamp) for single_stamp in raw_stamps_1])[0]
            idx_group = np.where([idx_stamp in single_group for single_group in groups_overlap_1])[0][0].item()
            shown_groups_1.append(idx_group)
        shown_groups_1 = np.unique(shown_groups_1)
        max_size_shown = 0
        for single_group in shown_groups_1:
            group_stamps = [raw_stamps_1[single_idx] for single_idx in groups_overlap_1[single_group]]
            group_stamps = np.stack(group_stamps, axis=0)
            group_size = group_stamps.shape[0]
            if group_size > max_size_shown:
                max_size_shown = group_size
            for j, single_stamp in enumerate(group_stamps):
                stamp_idx = int(1 * 1e4 + groups_overlap_1[single_group][j])
                color_for_display = viz.PALETTE['red']
                single_stamp = np.clip(single_stamp.copy(), a_min=page_start, a_max=page_end)  # new
                ax.plot(
                    single_stamp/fs, [valid_start-j*dy_valid, valid_start-j*dy_valid],
                    color=color_for_display, linewidth=1.5, label=valid_label)
                if (page_end - single_stamp[1])/fs > 0.5:  # new
                    ax.annotate(stamp_idx, (single_stamp[1]/fs+0.05, valid_start-j*dy_valid-10), fontsize=7)
                shown_valid = True
                valid_label = None
        valid_1_center = valid_start - (max_size_shown//2) * dy_valid
        # Show valid 2
        valid_start = - max_size_shown * dy_valid - 200
        shown_groups_2 = []
        for j, this_stamp in enumerate(segment_stamps_valid_2):
            idx_stamp = np.where([np.all(this_stamp == single_stamp) for single_stamp in raw_stamps_2])[0]
            idx_group = np.where([idx_stamp in single_group for single_group in groups_overlap_2])[0][0].item()
            shown_groups_2.append(idx_group)
        shown_groups_2 = np.unique(shown_groups_2)
        max_size_shown = 0
        for single_group in shown_groups_2:
            group_stamps = [raw_stamps_2[single_idx] for single_idx in groups_overlap_2[single_group]]
            group_stamps = np.stack(group_stamps, axis=0)
            group_size = group_stamps.shape[0]
            if group_size > max_size_shown:
                max_size_shown = group_size
            for j, single_stamp in enumerate(group_stamps):
                stamp_idx = int(2 * 1e4 + groups_overlap_2[single_group][j])
                color_for_display = viz.PALETTE['red']
                single_stamp = np.clip(single_stamp.copy(), a_min=page_start, a_max=page_end)  # new
                ax.plot(
                    single_stamp/fs, [valid_start-j*dy_valid, valid_start-j*dy_valid],
                    color=color_for_display, linewidth=1.5, label=valid_label)
                if (page_end - single_stamp[1])/fs > 0.5:  # new
                    ax.annotate(stamp_idx, (single_stamp[1]/fs+0.05, valid_start-j*dy_valid-10), fontsize=7)
                shown_valid = True
                valid_label = None
        valid_2_center = valid_start - (max_size_shown//2) * dy_valid
        # Signal
        y_max = 150
        y_sep = 300
        start_signal_plot = valid_start - max_size_shown * dy_valid - y_sep
        y_minor_ticks = []
        for k, name in enumerate(to_show_names):
            if name == 'F4-C4':
                stamp_center = start_signal_plot-y_sep*k
            #if name == 'EMG':
            #    continue
            segment_fs = fs
            segment_start = int(page_chosen * page_duration * segment_fs)
            segment_end = int(segment_start + page_duration * segment_fs)
            segment_signal = signal_dict[name][segment_start:segment_end]
            segment_time_axis = np.arange(segment_start, segment_end) / segment_fs
            ax.plot(
                segment_time_axis, start_signal_plot-y_sep*k + segment_signal, linewidth=0.7, color=viz.PALETTE['grey'])
            y_minor_ticks.append(start_signal_plot-y_sep*k + signal_uv_to_display)
            y_minor_ticks.append(start_signal_plot-y_sep*k - signal_uv_to_display)
        plotter.add_scalebar(
            ax, matchx=False, matchy=False, hidex=False, hidey=False, sizex=1, sizey=100,
            labelx='1 s', labely='100 uV', loc=1)
        expert_shown = False
        for expert_stamp in segment_stamps:
            expert_stamp = np.clip(expert_stamp.copy(), a_min=page_start, a_max=page_end)  # new
            label = None if expert_shown else 'Accepted mark (automatic)'
            ax.plot(
                expert_stamp / fs, [stamp_center-50, stamp_center-50],
                color=viz.PALETTE['green'], linewidth=2, label=label)
            expert_shown = True
        expert_manual_shown = False
        for final_stamp in segment_stamps_final:
            final_stamp = np.clip(final_stamp.copy(), a_min=page_start, a_max=page_end)  # new
            label = None if expert_manual_shown else 'Expert Final Version'
            ax.fill_between(
                final_stamp / fs, 100+stamp_center, -100+stamp_center,
                facecolor=viz.PALETTE['grey'], alpha=0.4,  label=label, edgecolor='k')
            expert_manual_shown = True
        ticks_valid = [valid_1_center, valid_2_center]
        ticks_signal = [start_signal_plot-y_sep*k for k in range(len(to_show_names))]
        ticklabels_valid = ['V1', 'V2']
        total_ticks = ticks_valid + ticks_signal
        total_ticklabels = ticklabels_valid + to_show_names[:-2] + ['MOR', 'EMG']
        ax.set_yticks(total_ticks)
        ax.set_yticklabels(total_ticklabels)
        ax.set_xlim([time_axis[0], time_axis[-1]])
        ax.set_ylim([-y_max - 30 + ticks_signal[-1], 100])
        ax.set_title('Subject %d (%s INTA). Page in record: %1.1f. (intervals of 0.5s are shown as a vertical grid).'
                     % (subject_id, NAMES[subject_id-1], page_chosen), fontsize=10, y=1.05)
        ax.set_xticks(x_ticks)
        ax.set_xticks(np.arange(time_axis[0], time_axis[-1], 0.5), minor=True)
        ax.grid(b=True, axis='x', which='minor')
        ax.tick_params(labelsize=7.5, labelbottom=True ,labeltop=True, bottom=True, top=True)
        ax.set_aspect(1/microvolt_per_second)
        ax.set_xlabel('Time [s]', fontsize=8)
        if expert_shown or shown_valid:
            lg = ax.legend(loc='lower left', fontsize=8)
            for lh in lg.legendHandles:
                lh.set_alpha(1.0)
        plt.tight_layout()
        return ax


    # -----------------------------
    # Save files for conflicts only
    # -----------------------------
    start_from_conflict = 1  # min is 1
    last_conflict = None  # None

    folder_name = '%s_conflicts' % NAMES[subject_id - 1]
    os.makedirs(folder_name, exist_ok=True)
    n_conflicts = conflict_pages.size
    if last_conflict is None:
        last_conflict = n_conflicts
    print('Total conflicting pages: %d' % n_conflicts)
    fig, ax = plt.subplots(1, 1, figsize=(12, 1+len(to_show_names)), dpi=180)
    for conflict_id in range(start_from_conflict, last_conflict + 1):
        fname = os.path.join(folder_name, 'conflict_%03d.pdf' % conflict_id)
        ax.clear()
        page_chosen = conflict_pages[conflict_id-1]
        ax = plot_page_conflict(page_chosen, ax)
        plt.savefig(fname, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close('all')

    if save_selected_pages:
        # -----------------------------
        # All SQ2 and SQ3 pages
        # -----------------------------
        original_page_duration = 30

        path_states = os.path.join(dataset_dir, 'label/state/', 'StagesOnly_%s.txt' % NAMES[subject_id - 1])
        states = np.loadtxt(path_states, dtype='i', delimiter=' ')
        # Crop signal and states to a valid length
        block_duration = 60
        block_size = block_duration * fs
        n_blocks = np.floor(signal_dict['F4-C4'].size / block_size)
        max_sample = int(n_blocks * block_size)
        max_page = int(max_sample / (original_page_duration * fs))
        hypnogram_original = states[:max_page]

        # Collect stages
        signal_total_duration = len(hypnogram_original) * original_page_duration
        select_pages_original = np.sort(np.concatenate([np.where(hypnogram_original == state_id)[0] for state_id in stages_valid]))
        print("Original selected pages: %d" % len(select_pages_original))
        onsets_original = select_pages_original * original_page_duration
        offsets_original = (select_pages_original + 1) * original_page_duration
        total_pages = int(np.ceil(signal_total_duration / page_duration))
        select_pages_onehot = np.zeros(total_pages, dtype=np.int16)
        for i in range(total_pages):
            onset_new_page = i * page_duration
            offset_new_page = (i + 1) * page_duration
            for j in range(select_pages_original.size):
                intersection = (onset_new_page < offsets_original[j]) and (onsets_original[j] < offset_new_page)
                if intersection:
                    select_pages_onehot[i] = 1
                    break
        select_pages = np.where(select_pages_onehot == 1)[0]
        print(select_pages.size)

        # save
        start_from_page = 1  # min is 1
        last_page = None  # None
        stages_str = [inta_stages_names[state_id] for state_id in stages_valid]
        stages_str.sort()
        stages_str = "-".join(stages_str)
        folder_name = '%s_%s' % (NAMES[subject_id - 1], stages_str)
        os.makedirs(folder_name, exist_ok=True)
        n_selected_pages = select_pages.size
        if last_page is None:
            last_page = n_selected_pages
        print('Total selected pages: %d' % n_selected_pages)
        print("Saving selected pages at %s" % folder_name)
        fig, ax = plt.subplots(1, 1, figsize=(12, 1+len(to_show_names)), dpi=180)
        for page_id in range(start_from_page, last_page + 1):
            fname = os.path.join(folder_name, 'page_%03d.pdf' % page_id)
            ax.clear()
            page_chosen = select_pages[page_id-1]
            ax = plot_page_conflict(page_chosen, ax)
            plt.savefig(fname, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close('all')
