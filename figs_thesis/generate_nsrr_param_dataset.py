import os
import sys
import pickle
import time

PROJECT_ROOT = os.path.abspath('..')
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants, viz, pkeys
from sleeprnn.data import utils
from figs_thesis import fig_utils

viz.notebook_full_width()

param_filtering_fn = fig_utils.get_filtered_signal_for_event
param_frequency_fn = fig_utils.get_frequency_by_fft
param_amplitude_fn = fig_utils.get_amplitude_event

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')


if __name__ == "__main__":
    nsrr = load_dataset(constants.NSRR_SS_NAME, load_checkpoint=True, params={pkeys.PAGE_DURATION: 30})

    # Load predictions
    parts_to_load = np.arange(12)
    pred_objects = []
    for part in parts_to_load:
        print("Loading part", part)
        filepath = os.path.join(
            RESULTS_PATH, 'predictions_nsrr_ss',
            'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
            'v2_time',
            'prediction_part%d.pkl' % part
        )
        with open(filepath, 'rb') as handle:
            pred_object = pickle.load(handle)
        pred_object.set_parent_dataset(nsrr)
        pred_objects.append(pred_object)

    # Filenames of dataset checkpoints
    byevent_ckpt_path = os.path.join(
        RESULTS_PATH, 'predictions_nsrr_ss',
        'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
        'v2_time',
        'table_byevent.csv'
    )
    bysubject_ckpt_path = os.path.join(
        RESULTS_PATH, 'predictions_nsrr_ss',
        'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
        'v2_time',
        'table_bysubject.csv'
    )

    # Perform computation and save checkpoint
    table_byevent = {
        'subject_id': [],
        'mark_id': [],
        'duration': [],
        'amplitude': [],
        'frequency': [],
    }
    table_bysubject = {
        'subject_id': [],
        'age': [],
        'female': [],
        'n2_minutes': [],
        'n2_abs_sigma_power': [],
        'n2_rel_sigma_power': [],
        'n2_pl_exponent': [],
        'density': [],
        'duration': [],
        'amplitude': [],
        'frequency': [],
    }

    min_n2_minutes = 60
    verbose_min_minutes = False

    start_time = time.time()
    print("Generating table of parameters")
    for part_id, predictions in enumerate(pred_objects):
        print("Processing Part %d / %d" % (part_id + 1, len(pred_objects)))
        for subject_id in tqdm(predictions.all_ids):
            n2_pages = predictions.data[subject_id]['n2_pages']
            n2_minutes = n2_pages.size * nsrr.original_page_duration / 60
            if n2_minutes < min_n2_minutes:
                if verbose_min_minutes:
                    print("Skipped by N2 minutes: Subject %s with %d N2 minutes" % (subject_id, n2_minutes))
                continue

            # Now compute parameters
            subject_data = nsrr.read_subject_data(subject_id, exclusion_of_pages=False)
            signal = subject_data['signal']
            age = float(subject_data['age'].item())
            female = int(subject_data['sex'].item() == 'f')

            # By-subject spectral parameters
            signal_n2 = signal.reshape(-1, nsrr.fs * nsrr.original_page_duration)[n2_pages].flatten()
            freq, power = utils.power_spectrum_by_sliding_window(signal_n2, nsrr.fs, window_duration=5)
            # a) Absolute sigma power
            n2_abs_sigma_power = power[(freq >= 11) & (freq <= 16)].mean()
            # b) Relative sigma power (as in Lacourse)
            n2_broad_power = power[(freq >= 4.5) & (freq <= 30)].mean()
            n2_rel_sigma_power = n2_abs_sigma_power / n2_broad_power
            # c) Power law exponent 2-30 Hz without sigma band (same as exclusion process)
            locs_to_use = np.where(((freq >= 2) & (freq < 10)) | ((freq > 17) & (freq <= 30)))[0]
            log_x = np.log(freq[locs_to_use])
            log_y = np.log(power[locs_to_use])
            n2_pl_exponent, _, _, _, _ = scipy.stats.linregress(log_x, log_y)

            # Spindle parameters
            marks = predictions.get_subject_stamps(subject_id, pages_subset='n2')
            n_marks = marks.shape[0]
            density = n_marks / n2_minutes  # epm
            if n_marks == 0:
                # Dummy entries
                be_subject_id = [subject_id]
                be_mark_id = [np.nan]
                be_duration = [np.nan]
                be_amplitude = [np.nan]
                be_frequency = [np.nan]
            else:
                be_subject_id = [subject_id] * n_marks
                be_mark_id = np.arange(n_marks)

                be_duration = (marks[:, 1] - marks[:, 0] + 1) / nsrr.fs

                filt_signal = param_filtering_fn(signal, nsrr.fs, constants.SPINDLE)
                signal_events = [filt_signal[e[0]:(e[1] + 1)] for e in marks]

                be_amplitude = np.array([
                    param_amplitude_fn(s, nsrr.fs, constants.SPINDLE) for s in signal_events
                ])

                be_frequency = np.array([
                    param_frequency_fn(s, nsrr.fs) for s in signal_events
                ])

            # By-subject averages
            bs_duration = np.mean(be_duration)
            bs_amplitude = np.mean(be_amplitude)
            bs_frequency = np.mean(be_frequency)

            # Save parameters
            table_byevent['subject_id'].append(be_subject_id)
            table_byevent['mark_id'].append(be_mark_id)
            table_byevent['duration'].append(be_duration)
            table_byevent['amplitude'].append(be_amplitude)
            table_byevent['frequency'].append(be_frequency)

            table_bysubject['subject_id'].append(subject_id)
            table_bysubject['age'].append(age)
            table_bysubject['female'].append(female)
            table_bysubject['n2_minutes'].append(n2_minutes)
            table_bysubject['n2_abs_sigma_power'].append(n2_abs_sigma_power)
            table_bysubject['n2_rel_sigma_power'].append(n2_rel_sigma_power)
            table_bysubject['n2_pl_exponent'].append(n2_pl_exponent)
            table_bysubject['density'].append(density)
            table_bysubject['duration'].append(bs_duration)
            table_bysubject['amplitude'].append(bs_amplitude)
            table_bysubject['frequency'].append(bs_frequency)
    end_time = time.time()
    et_time = (end_time - start_time) / 60  # minutes
    print("Elapsed time: %1.4f minutes" % et_time)

    for key in table_byevent:
        table_byevent[key] = np.concatenate(table_byevent[key])
    table_byevent = pd.DataFrame.from_dict(table_byevent)
    table_bysubject = pd.DataFrame.from_dict(table_bysubject)

    # Save checkpoint
    print("Saving checkpoint")
    table_byevent.to_csv(byevent_ckpt_path, index=False)
    table_bysubject.to_csv(bysubject_ckpt_path, index=False)
    print("Done.")
