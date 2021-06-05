from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats


if __name__ == '__main__':
    result_1 = [0.8428221288515406, 0.8121756380962502, 0.8378641244161973]
    result_2 = [0.8342261458355653, 0.8151154354749911, 0.8316107788455076]
    print("Result 1: %1.4f +- %1.4f (N=%d)" % (np.mean(result_1), np.std(result_1), len(result_1)))
    print("Result 2: %1.4f +- %1.4f (N=%d)" % (np.mean(result_2), np.std(result_2), len(result_2)))
    pvalue = stats.ttest_ind(result_1, result_2, equal_var=False)[1]
    print("P-value:", pvalue)
