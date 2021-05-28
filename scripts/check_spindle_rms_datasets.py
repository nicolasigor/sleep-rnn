from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import pkeys, constants


def get_frequency(signal, fs):
    signal2fft = np.concatenate([
        np.zeros(fs * 2),
        signal * np.hanning(signal.size),
        np.zeros(fs * 2)])
    y = np.fft.rfft(signal2fft)
    y = np.abs(y)
    f = np.fft.rfftfreq(signal2fft.size, d=1. / fs)
    y_sigma = y[(f >= 8) & (f <= 18)]
    f_sigma = f[(f >= 8) & (f <= 18)]
    max_loc = np.argmax(y_sigma)
    return f_sigma[max_loc]


if __name__ == '__main__':
    fs = 200
    which_expert = 1
    datasets_name_clip_value_list = [
        (constants.INTA_SS_NAME, 300),
        (constants.MODA_SS_NAME, 200),
        (constants.MASS_SS_NAME, 200),
    ]
    params = {pkeys.FS: fs}

    for dataset_name, clip_value in datasets_name_clip_value_list:
        name_to_save = "%s-e%d" % (dataset_name.split("_")[0], which_expert)
        print("\nDataset: %s" % name_to_save)
        dataset = load_dataset(dataset_name, load_checkpoint=True, params=params, verbose=False)
        all_ids = dataset.get_ids()
        all_rms = []
        all_freqs = []
        for subject_id in all_ids:
            signal = dataset.get_subject_signal(subject_id, which_expert=which_expert, normalize_clip=False)
            signal_filt = utils.broad_filter(signal, fs, 8, 18)
            marks = dataset.get_subject_stamps(subject_id, which_expert=which_expert, pages_subset=constants.N2_RECORD)
            spindles = [signal_filt[m[0]:m[1]+1] for m in marks]
            freqs = [get_frequency(s, fs) for s in spindles]
            rms = [np.sqrt(np.mean(s ** 2)) for s in spindles]
            all_rms.append(rms)
            all_freqs.append(freqs)
        all_rms = np.concatenate(all_rms)
        all_freqs = np.concatenate(all_freqs)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
        ax = axes[0]
        ax.hist(all_rms)
        ax.set_xlabel("Voltage ($\mu$V)")
        ax.set_title("%s (min: %1.2f, mean: %1.2f, prct5 %1.2f)" % (
            name_to_save,
            all_rms.min(),
            all_rms.mean(),
            np.percentile(all_rms, 5)
        ), fontsize=8)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xlim([0, 30])
        ax.tick_params(labelsize=8)

        ax = axes[1]
        ax.hist(all_freqs)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title("%s (min: %1.2f, mean: %1.2f, max %1.2f)" % (
            name_to_save,
            all_freqs.min(),
            all_freqs.mean(),
            all_freqs.max(),
        ), fontsize=8)
        ax.set_xticks([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        ax.set_xlim([8, 18])
        ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.show()



