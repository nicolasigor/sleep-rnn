from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import pyedflib

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.common import viz
from sleeprnn.data.utils import PATH_DATA
from sleeprnn.data.mass_ss import PATH_MASS_RELATIVE, PATH_REC, PATH_MARKS

if __name__ == '__main__':
    subject_id = 1
    file_rec = os.path.join(
        project_root, PATH_DATA, PATH_MASS_RELATIVE, PATH_REC,
        '01-02-%04d PSG.edf' % subject_id)
    file_ss = os.path.join(
        project_root, PATH_DATA, PATH_MASS_RELATIVE, PATH_MARKS,
        '01-02-%04d SpindleE1.edf' % subject_id)

    chosen_unipolar = [
        'F3',
        'C3',
        'P3',
        'Cz',
        'F4',
        'C4',
        'P4'
    ]
    chosen_bipolar = [
        ('F3', 'C3'),
        ('F4', 'C4')
    ]

    marked_ch = 'C3'

    unipolar_signals = {}
    bipolar_signals = {}

    with pyedflib.EdfReader(file_rec) as file:
        channel_names = file.getSignalLabels()
        for name in chosen_unipolar:
            format_name = 'EEG %s-CLE' % name
            channel_to_extract = channel_names.index(format_name)
            this_signal = file.readSignal(channel_to_extract)
            unipolar_signals[name] = this_signal
        for name in chosen_bipolar:
            bipolar_signals[name] = unipolar_signals[name[0]] - unipolar_signals[name[1]]

    # Show signals
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    ax.plot()



        #channel_to_extract = channel_names.index(self.channel)
        #signal = file.readSignal(channel_to_extract)
        #fs = file.samplefrequency(channel_to_extract)


