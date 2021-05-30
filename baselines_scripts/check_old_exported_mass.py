from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils


fs = 256
data = scipy.io.loadmat('/home/ntapia/projects/sleep-rnn/resources/datasets/mass_s1_fs_256.mat')
x = data['signal'].flatten()
n_samples = x.size
last_sample = n_samples // (fs * 5)
last_sample = int(last_sample * fs * 5)
x = x[:last_sample]

f, y = utils.power_spectrum_by_sliding_window(x, fs)
plt.loglog(f, np.abs(y))
plt.show()

print(x.max(), x.min())
