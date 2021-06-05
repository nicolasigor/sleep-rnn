from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats


if __name__ == '__main__':
    # result_1 = [0.8428221288515406, 0.8121756380962502, 0.8378641244161973]
    # result_2 = [0.8342261458355653, 0.8151154354749911, 0.8316107788455076]
    # print("Result 1: %1.4f +- %1.4f (N=%d)" % (np.mean(result_1), np.std(result_1), len(result_1)))
    # print("Result 2: %1.4f +- %1.4f (N=%d)" % (np.mean(result_2), np.std(result_2), len(result_2)))
    # pvalue = stats.ttest_ind(result_1, result_2, equal_var=False)[1]
    # print("P-value:", pvalue)
    #

    # By subject
    v2_time_wave0 = {
        3: [0.74524715, 0.73359073, 0.7896679],
        5: [0.75371901, 0.69658887, 0.77124183],
        7: [0.75656325, 0.7774962,  0.79018789],
        9: [0.85394737, 0.85620915, 0.84615385],
        10: [0.86288416, 0.84749578, 0.85648415],
        11: [0.78406709, 0.77916667, 0.79184248],
        14: [0.71053985, 0.71134021, 0.72882259],
        17: [0.8024819,  0.81414141, 0.82890542],
        18: [0.79690522, 0.82352941, 0.83102987],
        19: [0.79714286, 0.80851064, 0.7826087],
    }
    v2_time_wave1 = {
        3: [0.78498294, 0.76862745, 0.77857143],
        5: [0.77741935, 0.6984127,  0.77294686],
        7: [0.77732794, 0.77205507, 0.7808642],
        9: [0.8513599,  0.85434357, 0.84935065],
        10: [0.85202312, 0.84293785, 0.85319516],
        11: [0.76933423, 0.77241379, 0.77445652],
        14: [0.73017563, 0.73627557, 0.72708002],
        17: [0.81059863, 0.81681682, 0.82180294],
        18: [0.79826673, 0.8252516,  0.83520431],
        19: [0.79234973, 0.81039755, 0.7723036]
    }
    subject_ids = list(v2_time_wave0.keys())
    subject_ids.sort()
    for subject_id in subject_ids:
        result_1 = v2_time_wave0[subject_id]
        result_2 = v2_time_wave1[subject_id]
        pvalue = stats.ttest_ind(result_1, result_2, equal_var=False)[1]
        print("Subject %02d: %1.4f+-%1.4f to %1.4f+-%1.4f (delta %+1.4f) (P %1.4f)" % (
            subject_id,
            np.mean(result_1), np.std(result_1),
            np.mean(result_2), np.std(result_2),
            np.mean(result_2) - np.mean(result_1),
            pvalue
        ))
