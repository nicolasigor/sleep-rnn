import os
import xml.etree.ElementTree as ET

import numpy as np
import pyedflib


def extract_id(fname, is_annotation):
    # Remove extension
    fname = ".".join(fname.split(".")[:-1])
    if is_annotation:
        # Remove last tag
        fname = "-".join(fname.split("-")[:-1])
    return fname


def prepare_paths(edf_folder, annot_folder):
    """
    Assuming annot_folder is the one in the NSRR format,
    nomenclature of files is
    edf: subjectid.edf
    xml: subjectid-nsrr.xml
    """

    edf_files = os.listdir(edf_folder)
    edf_files = [f for f in edf_files if '.edf' in f]

    annot_files = os.listdir(annot_folder)
    annot_files = [f for f in annot_files if '.xml' in f]

    edf_ids = [extract_id(fname, False) for fname in edf_files]
    annot_ids = [extract_id(fname, True) for fname in annot_files]

    # Keep only IDs with both files
    common_ids = list(set(edf_ids).intersection(set(annot_ids)))
    common_ids.sort()

    paths_dict = {}
    for single_id in common_ids:
        edf_loc = edf_ids.index(single_id)
        annot_loc = annot_ids.index(single_id)
        paths_dict[single_id] = {
            'edf': os.path.join(edf_folder, edf_files[edf_loc]),
            'annot': os.path.join(annot_folder, annot_files[annot_loc])
        }
    return paths_dict


def read_hypnogram(annot_path):
    tree = ET.parse(annot_path)
    root = tree.getroot()
    scored_events = root.find('ScoredEvents')
    epoch_length = float(root.find("EpochLength").text)
    # print(ET.tostring(root, encoding='utf8').decode('utf8'))
    stage_labels = []
    stage_stamps = []
    for event in scored_events:
        e_type = event.find("EventType").text
        if e_type == "Stages|Stages":
            stage_name = event.find("EventConcept").text
            stage_start = float(event.find("Start").text)
            stage_duration = float(event.find("Duration").text)
            # Normalize variable-length epoch to a number of fixed-length epochs
            n_epochs = int(stage_duration / epoch_length)
            for i in range(n_epochs):
                stage_stamps.append([stage_start + epoch_length * i, epoch_length])
                stage_labels.append(stage_name)
    stage_labels = np.array(stage_labels)
    stage_stamps = np.stack(stage_stamps, axis=0)
    idx_sorted = np.argsort(stage_stamps[:, 0])
    stage_labels = stage_labels[idx_sorted]
    stage_stamps = stage_stamps[idx_sorted, :]
    stage_start_times = stage_stamps[:, 0].astype(np.float32)
    return stage_labels, stage_start_times, epoch_length


def get_edf_info(edf_path):
    fs_list = []
    with pyedflib.EdfReader(edf_path) as file:
        channel_names = file.getSignalLabels()
        for chn in channel_names:
            channel_to_extract = channel_names.index(chn)
            fs = file.samplefrequency(channel_to_extract)
            fs_list.append(fs)
    return channel_names, fs_list


def read_edf_channel(edf_path, channel_priority_list):
    with pyedflib.EdfReader(edf_path) as file:
        channel_names = file.getSignalLabels()

        channel_found = None
        for chn_pair in channel_priority_list:
            if np.all([chn in channel_names for chn in chn_pair]):
                channel_found = chn_pair
                break
        if channel_found is None:
            return None

        channel_to_extract = channel_names.index(channel_found[0])
        signal = file.readSignal(channel_to_extract)
        fs = file.samplefrequency(channel_to_extract)
        if len(channel_found) == 2:
            channel_to_extract = channel_names.index(channel_found[1])
            signal_2 = file.readSignal(channel_to_extract)
            fs_2 = file.samplefrequency(channel_to_extract)
            if fs != fs_2:
                return None
            signal = signal - signal_2
    return signal, fs, channel_found
