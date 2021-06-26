import os
import sys

import numpy as np

sys.path.append("..")

from nsrr import nsrr_utils

NSRR_PATH = os.path.abspath("~/Projects/Sleep_Databases/NSRR_Databases")


if __name__ == "__main__":
    print("Check SHHS1")
    edf_folder = os.path.join(NSRR_PATH, "shhs/polysomnography/edfs/shhs1")
    annot_folder = os.path.join(NSRR_PATH, "shhs/polysomnography/annotations-events-nsrr/shhs1")

    paths_dict = nsrr_utils.prepare_paths(edf_folder, annot_folder)
    subject_ids = list(paths_dict.keys())
    for subject_id in subject_ids:
        edf_path = paths_dict[subject_id]['edf']
        annot_path = paths_dict[subject_id]['annot']
        stage_labels, stage_start_times, epoch_length = nsrr_utils.read_hypnogram(annot_path)
        total_pages = (stage_start_times[-1] + epoch_length) / epoch_length
        print("Subject %s, hypno: %d labels, epochLength %s, first start %s, last start %s, required labels %s" % (
            subject_id, len(stage_labels), epoch_length, stage_start_times[0], stage_start_times[-1], total_pages
        ))
