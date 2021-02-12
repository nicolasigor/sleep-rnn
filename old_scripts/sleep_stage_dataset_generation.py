from sleeprnn.data.mass_ss import PATH_MASS_RELATIVE, PATH_STATES, PATH_REC
from sleeprnn.data.utils import PATH_DATA
import os
import pyedflib
import numpy as np
from sleeprnn.data import utils
from pprint import pprint


def read_eeg(path_eeg_file, channel_name, output_fs):
    """Loads signal from 'path_eeg_file', does filtering and resampling."""
    with pyedflib.EdfReader(path_eeg_file) as file:
        channel_names = file.getSignalLabels()
        channel_to_extract = channel_names.index(channel_name)
        signal = file.readSignal(channel_to_extract)
        fs_old = file.samplefrequency(channel_to_extract)
        # Check
        print('Channel extracted: %s' % file.getLabel(channel_to_extract))

    fs_old_round = int(np.round(fs_old))
    # Transform the original fs frequency with decimals to rounded version
    signal = utils.resample_signal_linear(
        signal, fs_old=fs_old, fs_new=fs_old_round)
    # Now resample to the required frequency
    if output_fs != fs_old_round:
        signal = utils.resample_signal(
            signal, fs_old=fs_old_round, fs_new=output_fs)
    signal = signal.astype(np.float32)
    return signal


def read_states(path_states_file, signal_length, page_duration, fs, state_ids, unknown_id):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        # Total pages not necessarily equal to total_annots
        total_pages = int(np.ceil(signal_length / (page_duration * fs)))

        with pyedflib.EdfReader(path_states_file) as file:
            annotations = file.readAnnotations()

        onsets = np.array(annotations[0])
        durations = np.round(np.array(annotations[1]))
        stages_str = annotations[2]
        # keep only 20s durations
        valid_idx = (durations == page_duration)
        onsets = onsets[valid_idx]
        onsets_pages = np.round(onsets / page_duration).astype(np.int32)
        stages_str = stages_str[valid_idx]
        stages_char = [single_annot[-1] for single_annot in stages_str]

        # Build complete hypnogram
        total_annots = len(stages_char)

        not_unkown_ids = [
            state_id for state_id in state_ids
            if state_id != unknown_id]
        not_unkown_state_dict = {}
        for state_id in not_unkown_ids:
            state_idx = np.where(
                [stages_char[i] == state_id for i in range(total_annots)])[0]
            not_unkown_state_dict[state_id] = onsets_pages[state_idx]
        hypnogram = []
        for page in range(total_pages):
            state_not_found = True
            for state_id in not_unkown_ids:
                if page in not_unkown_state_dict[state_id] and state_not_found:
                    hypnogram.append(state_id)
                    state_not_found = False
            if state_not_found:
                hypnogram.append(unknown_id)
        hypnogram = np.asarray(hypnogram)

        return hypnogram


if __name__ == '__main__':
    # subject_id = 1
    for subject_id in range(1, 20):
        file_signal = os.path.join(
            PATH_DATA, PATH_MASS_RELATIVE, PATH_REC, '01-02-%04d PSG.edf' % subject_id)

        file_state = os.path.join(
            PATH_DATA, PATH_MASS_RELATIVE, PATH_STATES, '01-02-%04d Base.edf' % subject_id)
        state_ids = np.array(['1', '2', '3', '4', 'R', 'W', '?'])
        unknown_id = '?'

        useful_channels = [
            'EEG F3-CLE',  # Frontal
            'EEG C3-CLE',  # Central
            'EEG O1-CLE',  # Occipital
            'EMG Chin',    # EMG
            'EOG Left Horiz',  # Left eye
            'EOG Right Horiz',  # Right eye
        ]

        output_fs = 200

        # Read signals
        signal_list = []
        for name in useful_channels:
            this_signal = read_eeg(file_signal, name, output_fs)
            signal_list.append(this_signal)

        [print(len(this_signal)) for this_signal in signal_list]



