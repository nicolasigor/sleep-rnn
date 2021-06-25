import os
from pprint import pprint
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, hilbert

project_root = '..'
sys.path.append(project_root)

from sleeprnn.common import viz, constants
from sleeprnn.helpers import reader, plotter, misc, performer
from sleeprnn.detection import metrics
from figs_thesis import fig_utils
from baselines_scripts.butils import get_partitions
from sleeprnn.detection.feeder_dataset import FeederDataset
from sklearn.linear_model import LinearRegression, HuberRegressor
from sleeprnn.data import utils

RESULTS_PATH = os.path.join(project_root, 'results')
BASELINES_PATH = os.path.join(project_root, 'resources', 'comparison_data', 'baselines_2021')

models = [constants.V2_TIME, constants.V2_CWT1D]
baselines_ss = ['dosed', 'a7']
baselines_kc = ['dosed', 'spinky']
print_model_names = {
    constants.V2_TIME: 'REDv2-Time',
    constants.V2_CWT1D: 'REDv2-CWT',
    'dosed': 'DOSED',
    'a7': 'A7',
    'spinky': 'Spinky'
}
print_dataset_names = {
    (constants.MASS_SS_NAME, 1): "MASS-SS2-E1SS",
    (constants.MASS_SS_NAME, 2): "MASS-SS2-E2SS",
    (constants.MASS_KC_NAME, 1): "MASS-SS2-KC",
    (constants.MODA_SS_NAME, 1): "MASS-MODA",
    (constants.INTA_SS_NAME, 1): "INTA-UCH",
}

eval_configs = [
    dict(dataset_name=constants.MASS_SS_NAME, expert=1, strategy='5cv', seeds=3),
    dict(dataset_name=constants.MASS_SS_NAME, expert=2, strategy='5cv', seeds=3),
    dict(dataset_name=constants.MASS_KC_NAME, expert=1, strategy='5cv', seeds=3),
    dict(dataset_name=constants.MODA_SS_NAME, expert=1, strategy='5cv', seeds=3),
    dict(dataset_name=constants.INTA_SS_NAME, expert=1, strategy='5cv', seeds=3),
]
metrics_list = []
for config in eval_configs:
    print("\nLoading", config)
    dataset = reader.load_dataset(config["dataset_name"], verbose=False)
    baselines = baselines_ss if dataset.event_name == constants.SPINDLE else baselines_kc

    # Collect predictions
    pred_dict = {}
    for model_version in models:
        tmp_dict = fig_utils.get_red_predictions(
            model_version, config["strategy"], dataset, config["expert"], verbose=False)
        # Retrieve only predictions, same format as baselines
        pred_dict[model_version] = {}
        for k in tmp_dict.keys():
            fold_subjects = tmp_dict[k][constants.TEST_SUBSET].all_ids
            fold_predictions = tmp_dict[k][constants.TEST_SUBSET].get_stamps()
            pred_dict[model_version][k] = {s: pred for s, pred in zip(fold_subjects, fold_predictions)}
    for baseline_name in baselines:
        pred_dict[baseline_name] = fig_utils.get_baseline_predictions(
            baseline_name, config["strategy"], config["dataset_name"], config["expert"])
print("Predictions collected.")

# Compute change due to threshold
adjusted_thr_list = np.arange(0.05, 0.95 + 0.001, 0.05)
metrics_curve_list = []  # [loc in config][model_name][fold_id][metric_name][loc in thr]
for config in eval_configs:
    average_mode = constants.MICRO_AVERAGE if (config["dataset_name"] == constants.MODA_SS_NAME) else constants.MACRO_AVERAGE
    print("\nLoading", config)
    dataset = reader.load_dataset(config["dataset_name"], verbose=False)
    baselines = baselines_ss if dataset.event_name == constants.SPINDLE else baselines_kc
    metrics_curve = {}
    for model_version in models:
        metrics_curve[model_version] = {}
        tmp_dict = fig_utils.get_red_predictions(model_version, config["strategy"], dataset, config["expert"], verbose=False)
        for k in tmp_dict.keys():
            optimal_thr = tmp_dict[k][constants.TEST_SUBSET].probability_threshold
            # print("Fold %d, optimal thr %1.3f" % (k, optimal_thr))
            # Get events
            fold_subjects = tmp_dict[k][constants.TEST_SUBSET].all_ids
            feed_d = FeederDataset(dataset, fold_subjects, constants.N2_RECORD, which_expert=config["expert"])
            events_list = feed_d.get_stamps()
            # Get predictions
            tmp_metric_dict_list = []
            for adjusted_thr in adjusted_thr_list:
                tmp_dict[k][constants.TEST_SUBSET].set_probability_threshold(adjusted_thr, adjusted_by_threshold=optimal_thr, verbose=False)
                detections_list = tmp_dict[k][constants.TEST_SUBSET].get_stamps()
                performance = fig_utils.compute_fold_performance(events_list, detections_list, average_mode)
                tmp_metric_dict_list.append(performance)
            # list of dict -> dict of list
            dict_of_list = {}
            for metric_key in tmp_metric_dict_list[0].keys():
                dict_of_list[metric_key] = np.array([tmp_metric_dict_list[thr_idx][metric_key] for thr_idx in range(len(adjusted_thr_list))])
            metrics_curve[model_version][k] = dict_of_list
    metrics_curve_list.append(metrics_curve)
print("Done.")


fname = 'pr_curve_ckpt.pkl'
# save checkpoint
with open(fname, 'wb') as handle:
    pickle.dump(metrics_curve_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
