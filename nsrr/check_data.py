import os
import sys

import numpy as np

sys.path.append("..")

from nsrr import nsrr_utils

NSRR_PATH = os.path.abspath("/home/ntapia/Projects/Sleep_Databases/NSRR_Databases")


if __name__ == "__main__":
    data_paths = {
        'shhs1': {
            'edf': os.path.join(NSRR_PATH, "shhs/polysomnography/edfs/shhs1"),
            'annot': os.path.join(NSRR_PATH, "shhs/polysomnography/annotations-events-nsrr/shhs1")
        },
        'mros1': {
            'edf': os.path.join(NSRR_PATH, "mros/polysomnography/edfs/visit1"),
            'annot': os.path.join(NSRR_PATH, "mros/polysomnography/annotations-events-nsrr/visit1")
        },
    }

    channel_priority_labels = [
        ("EEG(sec)",),  # C3-A2
        ("EEG",),  # C4-A1
        ("C3", "A2"),
        ("C3", "M2"),
        ("C3-A2",),
        ("C3-M2",),
        ("C4", "A1"),
        ("C4", "M1"),
        ("C4-A1",),
        ("C4-M1",),
    ]

    dataset_name = 'shhs1'

    print("Check %s" % dataset_name)
    edf_folder = data_paths[dataset_name]['edf']
    annot_folder = data_paths[dataset_name]['annot']
    print("Paths:")
    print(edf_folder)
    print(annot_folder)

    paths_dict = nsrr_utils.prepare_paths(edf_folder, annot_folder)
    subject_ids = list(paths_dict.keys())

    # Reduced subset
    subject_ids = subject_ids[:50]

    print("Retrieved subjects: %d" % len(subject_ids))

    epoch_length_list = []
    first_label_start_list = []
    channel_ids_list = []
    for subject_id in subject_ids:
        edf_path = paths_dict[subject_id]['edf']
        annot_path = paths_dict[subject_id]['annot']

        # Hypnogram info
        stage_labels, stage_start_times, epoch_length = nsrr_utils.read_hypnogram(annot_path)
        epoch_length_list.append(epoch_length)
        first_label_start_list.append(stage_start_times[0])

        total_pages = (stage_start_times[-1] + epoch_length) / epoch_length
        labeled_pages = len(stage_labels)
        if labeled_pages != total_pages:
            print("Subject %s, hypno: %d labels, epochLength %s, first start %s, last start %s, required labels %s" % (
                subject_id, len(stage_labels), epoch_length, stage_start_times[0], stage_start_times[-1], total_pages
            ))

        # Signal info
        channel_names, fs_list = nsrr_utils.get_edf_info(edf_path)
        channel_found = None
        for chn_pair in channel_priority_labels:
            if np.all([chn in channel_names for chn in chn_pair]):
                channel_found = chn_pair
                break
        if channel_found is None:
            print("Subject %s without valid channels. Full list:" % subject_id, channel_names)
        else:
            channel_loc_1 = channel_names.index(channel_found[0])
            channel_name_1 = channel_names[channel_loc_1]
            channel_fs_1 = fs_list[channel_loc_1]
            if len(channel_found) == 2:
                channel_loc_2 = channel_names.index(channel_found[1])
                channel_name_2 = channel_names[channel_loc_2]
                channel_fs_2 = fs_list[channel_loc_2]
            else:
                channel_name_2 = ''
                channel_fs_2 = ''
            channel_str = '%s-%s, fs %s-%s' % (channel_name_1, channel_name_2, channel_fs_1, channel_fs_2)
            channel_ids_list.append(channel_str)

    print("Epoch length:", np.unique(epoch_length_list))
    print("First start:", np.unique(first_label_start_list))
    print("Valid channels available:", np.unique(channel_ids_list))
