from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np

detector_path = '../../'
sys.path.append(detector_path)

from sleep.data_ops import norm_clip_eeg, extract_pages

PATH_SAMPLE_SIGNAL = '../demo_data/ind_signal.npy'
PATH_SAMPLE_PAGES = '../demo_data/ind_pages.npy'

if __name__ == '__main__':
    ind_signal = np.load(PATH_SAMPLE_SIGNAL)
    ind_pages = np.load(PATH_SAMPLE_PAGES)
    fs = 200
    page_size = 4000
    clip_value = 5
    norm_signal = norm_clip_eeg(
        ind_signal, ind_pages, page_size, clip_value=clip_value)

    # Plot only N2 data
    n2_data = extract_pages(norm_signal, ind_pages, page_size)
    n2_signal = np.concatenate(n2_data)
    fig = plt.figure()
    limit = clip_value
    plt.hist(n2_signal[np.abs(n2_signal) <= limit], bins=50, density=True)
    plt.xlim([-limit, limit])
    plt.show()
