import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
sys.path.append("..")
from figs_thesis.fig_utils import get_frequency_by_zero_crossings

fs = 200

fc = 11
phase = np.pi/4

t = np.arange(2 * fs) / fs
x = np.cos(2 * np.pi * t * fc + phase)

# Densify
upsampling_factor = 100
fs_original = 1.0 / (t[1] - t[0])
fs_new = fs_original * upsampling_factor
t_new = np.arange(t[0], t[-1], 1.0 / fs_new)
x_new = interp1d(t, x)(t_new)
# Find frequency by zero crossing
# pos = x_new > 0
# zero_crossing_1 = (pos[:-1] & ~pos[1:]).nonzero()[0]
# npos = ~pos
# zero_crossing_2 = (npos[:-1] & ~npos[1:]).nonzero()[0]
# periods_1 = np.diff(zero_crossing_1)
# periods_2 = np.diff(zero_crossing_2)
# mean_period = np.concatenate([periods_1, periods_2]).mean()
# fc_estimated = upsampling_factor * fs / mean_period
pos = x_new > 0
npos = ~pos
zero_crossings = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
mean_inter_crossing_samples = np.diff(zero_crossings).mean()
fc_estimated = fs_new / (2.0 * mean_inter_crossing_samples)
print("Real frequency:", fc)
print("Estimated frequency:", fc_estimated)
print("Estimated frequency method:", get_frequency_by_zero_crossings(x, fs, use_median=True))

plt.plot(t, x, marker='.')
plt.plot(t_new[zero_crossings], x_new[zero_crossings], marker='o', linestyle="None")
# plt.plot(t_new[zero_crossing_2], x_new[zero_crossing_2], marker='o', linestyle="None")
plt.ylim([-0.01, 0.01])
plt.grid()
plt.show()
