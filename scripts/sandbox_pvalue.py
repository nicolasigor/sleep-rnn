from scipy.stats import ttest_ind_from_stats

baseline = { # dosed on moda
    "f1": (77.5, 1.7, 15),
    "recall": (76.4, 2.8, 15),
    "precision": (78.9, 3.0, 15),
    "miou": (71.4, 1.1, 15),
}

proposed = { # seed 10% moda from mass2
    "f1": (78.8, 1.5, 15), #(79.5, 2.4, 15),
    "recall": (78.8, 6.0, 15),
    "precision": (80.7, 3.6, 15),
    "miou": (82.1, 0.8, 15),
}

for metric_name in proposed.keys():
    pvalue = ttest_ind_from_stats(
        proposed[metric_name][0],
        proposed[metric_name][1],
        proposed[metric_name][2],
        baseline[metric_name][0],
        baseline[metric_name][1],
        baseline[metric_name][2],
        equal_var=False
    )[1]
    print(f"For metric {metric_name}, pvalue is {pvalue:.3f}")
