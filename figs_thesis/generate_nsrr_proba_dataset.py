import os
import sys
import pickle
import time

PROJECT_ROOT = os.path.abspath('..')
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants, viz, pkeys
from sleeprnn.data import utils
from sleeprnn.detection import det_utils
from figs_thesis import fig_utils

viz.notebook_full_width()

param_filtering_fn = fig_utils.get_filtered_signal_for_event
param_frequency_fn = fig_utils.get_frequency_by_fft
param_amplitude_fn = fig_utils.get_amplitude_event

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')


def load_predictions(parts_to_load, dataset, thr=0.5, verbose=True):
    if thr == 0.5:
        extra_str = ''
    else:
        extra_str = '_%1.2f' % thr
    pred_objects = []
    for part in parts_to_load:
        filepath = os.path.join(
            RESULTS_PATH, 'predictions_nsrr_ss',
            'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
            'v2_time',
            'prediction%s_part%d.pkl' % (extra_str, part)
        )
        with open(filepath, 'rb') as handle:
            pred_object = pickle.load(handle)
        pred_object.set_parent_dataset(dataset)
        pred_objects.append(pred_object)
    return pred_objects


def extract_pages_for_stamps_strict(stamps, pages_indices, page_size):
    """Returns stamps that are at completely contained on pages."""
    stamps_start_page = np.floor(stamps[:, 0] / page_size)
    stamps_end_page = np.floor(stamps[:, 1] / page_size)
    useful_idx = np.where(
        np.isin(stamps_start_page, pages_indices) & np.isin(stamps_end_page, pages_indices)
    )[0]
    pages_data = stamps[useful_idx, :]
    return pages_data


if __name__ == "__main__":

    # Load predictions
    parts_to_load = np.arange(12)

    nsrr = load_dataset(constants.NSRR_SS_NAME, load_checkpoint=True, params={pkeys.PAGE_DURATION: 30})
    pred_objects_1 = load_predictions(parts_to_load, nsrr)
    pred_objects_0 = load_predictions(parts_to_load, nsrr, thr=0.25)

    # Filenames of dataset checkpoints
    byevent_proba_ckpt_path = os.path.join(
        RESULTS_PATH, 'predictions_nsrr_ss',
        'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
        'v2_time',
        'table_byevent_proba.csv'
    )

    # Perform computation and save checkpoint
    bands_for_mean_power = [
        (0, 2),
        (2, 4),
        (4, 8),
        (8, 10),
        (11, 16),
        (16, 30),
        (4.5, 30),
    ]

    table_byevent_proba = {
        'subject_id': [],
        'age': [],
        'female': [],
        'center_sample': [],
        'prediction_part': [],
        'category': [],
        'probability': [],
        'duration': [],
        'frequency': [],
        'amplitude_pp': [],
        'amplitude_rms': [],
        'correlation': [],
        'covariance': [],
        'c10_density_real': [],
        'c10_density_all': [],
        'c10_abs_sigma_power': [],
        'c10_rel_sigma_power': [],
        'c10_abs_sigma_power_masked': [],
        'c10_rel_sigma_power_masked': [],
        'c20_density_real': [],
        'c20_density_all': [],
        'c20_abs_sigma_power': [],
        'c20_rel_sigma_power': [],
        'c20_abs_sigma_power_masked': [],
        'c20_rel_sigma_power_masked': [],
    }
    table_byevent_proba.update(
        {'mean_power_%s_%s' % band: [] for band in bands_for_mean_power}
    )

    min_n2_minutes = 60
    verbose_min_minutes = False

    start_time = time.time()
    print("Generating table of parameters")
    n_parts = len(pred_objects_1)
    for part_id in range(n_parts):
        predictions_1 = pred_objects_1[part_id]
        predictions_0 = pred_objects_0[part_id]
        print("Processing Part %d / %d" % (part_id + 1, n_parts))

        n_subjects = len(predictions_1.all_ids)

        for i_subject in tqdm(range(n_subjects)):
            subject_id = predictions_1.all_ids[i_subject]
            n2_pages = predictions_1.data[subject_id]['n2_pages']
            n2_minutes = n2_pages.size * nsrr.original_page_duration / 60
            if n2_minutes < min_n2_minutes:
                if verbose_min_minutes:
                    print("Skipped by N2 minutes: Subject %s with %d N2 minutes" % (subject_id, n2_minutes))
                continue

            marks_1 = predictions_1.get_subject_stamps(subject_id)  # Class 1 spindles (real)
            marks_0 = predictions_0.get_subject_stamps(subject_id)  # Class 0 "spindles" (false)
            # Let only those class 0 without intersecting class 1
            # If marks_1.size = 0 then marks_0 is by definition not intersecting
            if marks_1.size > 0:
                ov_mat = utils.get_overlap_matrix(marks_0, marks_1)
                is_intersecting = ov_mat.sum(axis=1)
                marks_0 = marks_0[is_intersecting == 0]
            if (marks_1.size + marks_0.size) == 0:
                continue  # There are no marks to work with

            # Now only keep N2 stage marks
            n2_pages = predictions_1.data[subject_id]['n2_pages']
            page_size = int(nsrr.fs * nsrr.original_page_duration)
            if marks_1.size > 0:
                marks_1 = extract_pages_for_stamps_strict(marks_1, n2_pages, page_size)
            if marks_0.size > 0:
                marks_0 = extract_pages_for_stamps_strict(marks_0, n2_pages, page_size)
            if (marks_1.size + marks_0.size) == 0:
                continue  # There are no marks to work with

            marks = []
            marks_class = []
            if marks_1.size > 0:
                marks.append(marks_1)
                marks_class.append([1] * marks_1.shape[0])
            if marks_0.size > 0:
                marks.append(marks_0)
                marks_class.append([0] * marks_0.shape[0])
            marks = np.concatenate(marks, axis=0).astype(np.int32)
            marks_class = np.concatenate(marks_class).astype(np.int32)
            n_marks = marks.shape[0]

            # Extract proba
            subject_proba = predictions_1.get_subject_probabilities(subject_id, return_adjusted=False)
            marks_proba = det_utils.get_event_probabilities(marks, subject_proba, downsampling_factor=8, proba_prc=75)
            marks_proba = marks_proba.astype(np.float32)
            # Extract signal
            subject_data = nsrr.read_subject_data(subject_id, exclusion_of_pages=False)
            signal = subject_data['signal'].astype(np.float64)
            age = float(subject_data['age'].item())
            female = int(subject_data['sex'].item() == 'f')

            # Parameters
            be_duration = (marks[:, 1] - marks[:, 0] + 1) / nsrr.fs

            filt_signal = param_filtering_fn(signal, nsrr.fs, constants.SPINDLE).astype(np.float64)
            signal_events = [filt_signal[e[0]:(e[1] + 1)] for e in marks]

            be_amplitude_pp = np.array([
                param_amplitude_fn(s, nsrr.fs, constants.SPINDLE) for s in signal_events
            ])

            be_amplitude_rms = np.array([
                np.sqrt(np.mean(s ** 2)) for s in signal_events
            ])

            be_frequency = np.array([
                param_frequency_fn(s, nsrr.fs) for s in signal_events
            ])

            # New parameters
            signal_raw_events = [signal[e[0]:(e[1] + 1)] for e in marks]

            # Measure mean power
            for band in bands_for_mean_power:
                table_byevent_proba['mean_power_%s_%s' % band].append([])
            for s in signal_raw_events:
                freq, power = fig_utils.get_fft_spectrum(s, nsrr.fs, pad_to_duration=10, f_min=0, f_max=30,
                                                         apply_hann_window=False)
                for band in bands_for_mean_power:
                    power_in_band = power[(freq >= band[0]) & (freq <= band[1])].mean()
                    table_byevent_proba['mean_power_%s_%s' % band][-1].append(power_in_band)
            for band in bands_for_mean_power:
                table_byevent_proba['mean_power_%s_%s' % band][-1] = np.array(
                    table_byevent_proba['mean_power_%s_%s' % band][-1], dtype=np.float32)

            # Covariance and correlation between sigma band and broad band
            cov_l = []
            corr_l = []
            window_size = int(0.3 * nsrr.fs)
            step_size = int(0.1 * nsrr.fs)
            for s, filt_s in zip(signal_raw_events, signal_events):
                duration_real = s.size
                duration_adjusted = np.clip(duration_real // step_size, a_min=window_size, a_max=None)
                n_windows = 1 + (duration_adjusted - window_size) // step_size
                s_stack = []
                filt_s_stack = []
                for i_win in range(n_windows):
                    start_win = i_win * step_size
                    end_win = start_win + window_size
                    s_stack.append(s[start_win:end_win])
                    filt_s_stack.append(filt_s[start_win:end_win])
                s_stack = np.stack(s_stack, axis=0)
                filt_s_stack = np.stack(filt_s_stack, axis=0)
                # remove mean
                s_stack = s_stack - s_stack.mean(axis=1).reshape(-1, 1)
                filt_s_stack = filt_s_stack - filt_s_stack.mean(axis=1).reshape(-1, 1)
                # variance and covariance
                var_s = (s_stack ** 2).mean(axis=1)
                var_f = (filt_s_stack ** 2).mean(axis=1)
                cov_sf = np.mean(s_stack * filt_s_stack, axis=1)
                # compute final stats
                cov = cov_sf.mean()
                corr = np.mean(cov_sf / np.sqrt(var_s * var_f))

                # s = s - s.mean()
                # filt_s = filt_s - filt_s.mean()
                # covariance
                # cov = np.mean(s * filt_s)
                cov_l.append(cov)
                # correlation
                # corr = np.corrcoef(s, filt_s)[0, 1]
                corr_l.append(corr)

            cov_l = np.array(cov_l, dtype=np.float32)
            corr_l = np.array(corr_l, dtype=np.float32)

            # Local stuff
            context_params = {
                'c10_density_real': [],
                'c10_density_all': [],
                'c10_abs_sigma_power': [],
                'c10_rel_sigma_power': [],
                'c10_abs_sigma_power_masked': [],
                'c10_rel_sigma_power_masked': [],
                'c20_density_real': [],
                'c20_density_all': [],
                'c20_abs_sigma_power': [],
                'c20_rel_sigma_power': [],
                'c20_abs_sigma_power_masked': [],
                'c20_rel_sigma_power_masked': [],
            }
            window_durations = [10, 20]
            for i_mark, mark in enumerate(marks):
                central_sample = mark.mean()
                for window_duration in window_durations:
                    window_size = int(window_duration * nsrr.fs)
                    start_sample = int(central_sample - window_size // 2)
                    end_sample = start_sample + window_size
                    # Local number of marks, by category
                    local_nmarks_real = utils.filter_stamps(marks_1, start_sample, end_sample).shape[0]
                    local_nmarks_both = utils.filter_stamps(marks, start_sample, end_sample).shape[0]
                    # Local sigma activity
                    segment_signal = signal[start_sample:end_sample]

                    # including event
                    freq, power = utils.power_spectrum_by_sliding_window(
                        segment_signal, nsrr.fs, window_duration=5)
                    # a) Absolute sigma power
                    local_abs_sigma_power = power[(freq >= 11) & (freq <= 16)].mean()
                    # b) Relative sigma power (as in Lacourse)
                    local_broad_power = power[(freq >= 4.5) & (freq <= 30)].mean()
                    local_rel_sigma_power = local_abs_sigma_power / local_broad_power

                    # masking event
                    local_start = mark[0] - start_sample
                    local_end = mark[1] - start_sample
                    segment_signal_masked = segment_signal.copy()
                    segment_signal_masked[local_start:local_end] = 0
                    freq, power = utils.power_spectrum_by_sliding_window(
                        segment_signal_masked, nsrr.fs, window_duration=5)
                    # a) Absolute sigma power
                    local_mask_abs_sigma_power = power[(freq >= 11) & (freq <= 16)].mean()
                    # b) Relative sigma power (as in Lacourse)
                    local_mask_broad_power = power[(freq >= 4.5) & (freq <= 30)].mean()
                    local_mask_rel_sigma_power = local_mask_abs_sigma_power / local_mask_broad_power

                    # Save
                    context_params['c%d_density_real' % window_duration].append(local_nmarks_real)
                    context_params['c%d_density_all' % window_duration].append(local_nmarks_both)
                    context_params['c%d_abs_sigma_power' % window_duration].append(local_abs_sigma_power)
                    context_params['c%d_rel_sigma_power' % window_duration].append(local_rel_sigma_power)
                    context_params['c%d_abs_sigma_power_masked' % window_duration].append(local_mask_abs_sigma_power)
                    context_params['c%d_rel_sigma_power_masked' % window_duration].append(local_mask_rel_sigma_power)

            for key in context_params.keys():
                context_params[key] = np.array(context_params[key], dtype=np.float32)

            # New parameters
            table_byevent_proba['subject_id'].append([subject_id] * n_marks)
            table_byevent_proba['age'].append(np.array([age] * n_marks, dtype=np.float32))
            table_byevent_proba['female'].append(np.array([female] * n_marks, dtype=np.int32))
            table_byevent_proba['center_sample'].append(marks.mean(axis=1).astype(np.int32))
            table_byevent_proba['prediction_part'].append(np.array([part_id] * n_marks, dtype=np.int32))
            table_byevent_proba['category'].append(marks_class)
            table_byevent_proba['probability'].append(marks_proba.astype(np.float32))
            table_byevent_proba['duration'].append(be_duration.astype(np.float32))
            table_byevent_proba['frequency'].append(be_frequency.astype(np.float32))
            table_byevent_proba['amplitude_pp'].append(be_amplitude_pp.astype(np.float32))
            table_byevent_proba['amplitude_rms'].append(be_amplitude_rms.astype(np.float32))
            table_byevent_proba['covariance'].append(cov_l)
            table_byevent_proba['correlation'].append(corr_l)
            for key in context_params.keys():
                table_byevent_proba[key].append(context_params[key])

    for key in table_byevent_proba:
        table_byevent_proba[key] = np.concatenate(table_byevent_proba[key])
    table_byevent_proba = pd.DataFrame.from_dict(table_byevent_proba)

    # compute relative powers
    powers = table_byevent_proba[
        [col for col in table_byevent_proba.columns if 'mean_power' in col and "11_16" not in col]]
    powers = powers.div(table_byevent_proba["mean_power_11_16"], axis=0)
    powers = 1.0 / powers
    powers = powers.add_prefix("mean_power_11_16_relto_")
    table_byevent_proba = table_byevent_proba.merge(powers, left_index=True, right_index=True)

    # Save checkpoint
    print("Saving checkpoint")
    table_byevent_proba.to_csv(byevent_proba_ckpt_path, index=False)
    print("Done.")
