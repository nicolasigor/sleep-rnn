from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pprint import pprint

import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    my_data = np.load(
       '/home/ntapia/Projects/GitNico/sleep-rnn/results/lrp_dataset/20190713_v11_64_128_256_data_s06.npz')
    print('Arrays in NPZ')
    print(my_data.files)

    x = my_data['x']
    y = my_data['y']
    # cwt = my_data['cwt']
    pages = my_data['pages']
    stamps = my_data['stamps']
    predicted_y = my_data['predicted_y']

    which_stamp = 10

    page_size = 4000
    fs = 200
    context = 2
    times_stamp = stamps[which_stamp, :]
    which_page = np.where(pages == int(times_stamp[0] / page_size))[0][0]
    start_time = int((times_stamp[0] % page_size) - context * fs)
    end_time = int((times_stamp[1] % page_size) + context * fs)

    time_axis_long = np.arange(page_size) / fs
    time_axis_short = time_axis_long[::8]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=100)
    ax[0].plot(
        time_axis_long[start_time:end_time],
        x[which_page, (1000 + start_time):(1000+end_time)],
        linewidth=1.5, label='signal'
    )
    ax[0].set_xlim([time_axis_long[start_time], time_axis_long[end_time]])
    ax[0].legend(loc='upper right')

    ax[1].plot(
        time_axis_short[start_time//8:end_time//8],
        y[which_page, start_time//8:end_time//8],
        label='label'
    )
    ax[1].set_xlim([time_axis_long[start_time], time_axis_long[end_time]])
    ax[1].legend(loc='upper right')

    ax[2].plot(
        time_axis_short[start_time // 8:end_time // 8],
        predicted_y[which_page, start_time // 8:end_time // 8],
        label='prediction'
    )
    ax[2].set_xlim([time_axis_long[start_time], time_axis_long[end_time]])
    ax[2].legend(loc='upper right')

    # ax[3].imshow(
    #     cwt[which_page, start_time // 2:end_time // 2, :, 0].T, label='cwt-mag',
    #     interpolation='none', aspect='auto'
    # )
    #
    # ax[4].imshow(
    #     cwt[which_page, start_time // 2:end_time // 2, :, 1].T, label='cwt-pha',
    #     interpolation='none', aspect='auto'
    # )
    # phase = cwt[which_page, start_time // 2:end_time // 2, :, 1]

    plt.show()
