import os
import sys
import time

import numpy as np

project_root = ".."
sys.path.append(project_root)

from nsrr import nsrr_utils
from nsrr.nsrr_utils import NSRR_DATA_PATHS, CHANNEL_PRIORITY_LABELS
from sleeprnn.data import utils


DATASETS_PATH = os.path.join(project_root, 'resources', 'datasets', 'nsrr')


if __name__ == "__main__":

    keep_only_n2 = True
    dataset_name_list = [
        'chat1',
    ]
    reduced_number_of_subjects = None

    # ##################
    # AUXILIARY VARIABLES

    unknown_stage_label = "?"
    n2_id = 'Stage 2 sleep|2'
    target_fs = 200  # Hz

    # ##################
    # READ

    for dataset_name in dataset_name_list:

        save_dir = os.path.abspath(os.path.join(DATASETS_PATH, dataset_name, 'register_and_state'))
        os.makedirs(save_dir, exist_ok=True)

        print("\nReading %s" % dataset_name)
        edf_folder = NSRR_DATA_PATHS[dataset_name]['edf']
        annot_folder = NSRR_DATA_PATHS[dataset_name]['annot']
        print("From paths:")
        print(edf_folder)
        print(annot_folder)
        paths_dict = nsrr_utils.prepare_paths(edf_folder, annot_folder)
        subject_ids = list(paths_dict.keys())

        if reduced_number_of_subjects is not None:
            # Reduced subset
            subject_ids = subject_ids[:reduced_number_of_subjects]

        n_subjects = len(subject_ids)
        print("Retrieved subjects: %d" % n_subjects)
        print("Preprocessed files will be saved at %s" % save_dir)

        start_time = time.time()
        for i_sub, subject_id in enumerate(subject_ids):
            print("\nProcessing subject %s (%04d/%d)" % (subject_id, i_sub + 1, n_subjects))

            # Read data
            stage_labels, stage_start_times, epoch_length = nsrr_utils.read_hypnogram(
                paths_dict[subject_id]['annot'])
            signal, fs, channel_found = nsrr_utils.read_edf_channel(
                paths_dict[subject_id]['edf'], CHANNEL_PRIORITY_LABELS)

            # Channel id
            channel_id = " minus ".join(channel_found)

            # Filter and resample
            original_sampling_rate = fs

            # Transform the original fs frequency with decimals to rounded version if necessary
            fs_round = int(np.round(fs))
            if np.abs(fs_round - fs) > 1e-8:
                print("Linear interpolation from %s Hz to %s Hz" % (fs, fs_round))
                signal = utils.resample_signal_linear(signal, fs_old=fs, fs_new=fs_round)
            fs = fs_round

            # Broad bandpass filter to signal
            signal = utils.broad_filter(signal, fs, lowcut=0.1, highcut=35)
            # Now resample to the required frequency
            if fs != target_fs:
                print('Resampling channel %s from %s Hz to required %s Hz' % (channel_id, fs, target_fs))
                signal = utils.resample_signal(signal, fs_old=fs, fs_new=target_fs)
                resample_method = 'scipy.signal.resample_poly'
            else:
                print('Signal channel %s already at required %s Hz' % (channel_id, target_fs))
                resample_method = 'none'
            fs = target_fs

            # Ensure first label starts at t = 0
            valid_start_sample = int(stage_start_times[0] * fs)
            signal = signal[valid_start_sample:]
            stage_start_times = stage_start_times - stage_start_times[0]

            # Fill hypnogram if necessary
            hypno_total_pages = int((stage_start_times[-1] + epoch_length) / epoch_length)
            hypnogram = np.array([unknown_stage_label] * hypno_total_pages, dtype=stage_labels.dtype)
            labeled_locs = (stage_start_times / epoch_length).astype(np.int32)
            hypnogram[labeled_locs] = stage_labels

            # Ensure that both the signal and the hypnogram end at the same time
            epoch_samples = int(epoch_length * fs)
            signal_total_full_pages = int(signal.size // epoch_samples)
            valid_total_pages = min(hypno_total_pages, signal_total_full_pages)
            valid_total_samples = int(valid_total_pages * epoch_samples)
            hypnogram = hypnogram[:valid_total_pages]
            signal = signal[:valid_total_samples]

            if keep_only_n2:
                epoch_samples = int(epoch_length * fs)
                signal, hypnogram = nsrr_utils.short_signal_to_n2(signal, hypnogram, epoch_samples, n2_id)

            # Save subject data
            subject_data_dict = {
                'dataset': dataset_name,
                'subject_id': subject_id,
                'channel': channel_id,
                'signal': signal.astype(np.float32),
                'sampling_rate': fs,
                'hypnogram': hypnogram,
                'epoch_duration': epoch_length,
                'bandpass_filter': 'scipy.signal.butter, 0.1-35Hz, order 3',
                'resampling_function': resample_method,
                'original_sampling_rate': original_sampling_rate,
            }
            fpath = os.path.join(save_dir, '%s.npz' % subject_id)
            np.savez(fpath, **subject_data_dict)

            elapsed_time = time.time()-start_time
            print("E.T. %1.4f [s]" % elapsed_time)
