from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath('..')
sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np

from sleeprnn.data.utils import extract_pages_from_centers

if __name__ == "__main__":
    x = np.ones(100)
    t = np.arange(150)
    x[10:40] = -1
    x[60:80] = 2
    page_size = 20
    border_size = 2
    centers = [3, 15, 45, 90]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
    ax.plot(t[:100], x)
    for center in centers:
        ax.plot(center, x[center], marker='o')
    ax.set_ylim([-1.1, 2.1])
    plt.tight_layout()
    plt.show()

    segments = extract_pages_from_centers(x, centers, page_size, border_size)
    segments_t = extract_pages_from_centers(t, centers, page_size, border_size)
    half = segments[0].size // 2
    for i in range(len(centers)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
        ax.plot(segments_t[i], segments[i], marker='o')
        ax.plot(centers[i], x[centers[i]], marker='o')
        ax.plot(segments_t[i][half], segments[i][half], marker='s')
        ax.set_ylim([-1.1, 2.1])
        plt.tight_layout()
        plt.show()
