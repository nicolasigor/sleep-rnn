import os
from pprint import pprint
import sys

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
from sleeprnn.detection import metrics, det_utils
from figs_thesis import fig_utils
from baselines_scripts.butils import get_partitions
from sleeprnn.detection.feeder_dataset import FeederDataset
from sklearn.linear_model import LinearRegression, HuberRegressor
from sleeprnn.data import utils

RESULTS_PATH = os.path.join(project_root, 'results')
BASELINES_PATH = os.path.join(project_root, 'resources', 'comparison_data', 'baselines_2021')


if __name__ == '__main__':

    models = [constants.V2_TIME, constants.V2_CWT1D]
    print_model_names = {
        constants.V2_TIME: 'REDv2-Time',
        constants.V2_CWT1D: 'REDv2-CWT'
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
        # dict(dataset_name=constants.MASS_SS_NAME, expert=2, strategy='5cv', seeds=3),
        # dict(dataset_name=constants.MASS_KC_NAME, expert=1, strategy='5cv', seeds=3),
        # dict(dataset_name=constants.MODA_SS_NAME, expert=1, strategy='5cv', seeds=3),
        # dict(dataset_name=constants.INTA_SS_NAME, expert=1, strategy='5cv', seeds=3),
    ]
    metrics_list = []
    metrics_raw_list = []
    for config in eval_configs:
        print("\nLoading", config)
        dataset = reader.load_dataset(config["dataset_name"], verbose=False)
        _, _, test_ids_list = get_partitions(dataset, config["strategy"], config["seeds"])
        n_folds = len(test_ids_list)
        average_mode = constants.MICRO_AVERAGE if (
                    config["dataset_name"] == constants.MODA_SS_NAME) else constants.MACRO_AVERAGE
        # Collect predictions
        pred_dict = {}
        for model_version in models:
            tmp_dict = fig_utils.get_red_predictions(
                model_version, config["strategy"], dataset, config["expert"], verbose=False)
            # Retrieve probas and stamps
            pred_dict[model_version] = {s: {} for s in dataset.all_ids}
            for k in range(n_folds):
                fold_subjects = tmp_dict[k][constants.TEST_SUBSET].all_ids
                fold_probas = tmp_dict[k][constants.TEST_SUBSET].get_probabilities(return_adjusted=True)
                fold_stamps = tmp_dict[k][constants.TEST_SUBSET].get_stamps()
                for s, proba, stamp in zip(fold_subjects, fold_probas, fold_stamps):
                    pred_dict[model_version][s][k] = {'probability': proba, 'stamp': stamp}
                    # Generate typical dict
        pred_dict_original = {}
        for model_version in models:
            pred_dict_original[model_version] = {}
            for k in range(n_folds):
                pred_dict_original[model_version][k] = {s: pred_dict[model_version][s][k] for s in test_ids_list[k]}
        # Generate surrogate model
        # Random permutation of fold assignments of predictions
        pred_dict_permuted = {}
        for model_version in models:
            pred_dict_permuted[model_version] = {}
            for i_sub, subject_id in enumerate(dataset.all_ids):
                byfold_preds = pred_dict[model_version][subject_id]
                subject_folds = list(byfold_preds.keys())
                subject_preds = [byfold_preds[k] for k in subject_folds]
                subject_folds = np.random.RandomState(seed=i_sub).permutation(subject_folds)
                pred_dict_permuted[model_version][subject_id] = {k: pred for k, pred in
                                                                 zip(subject_folds, subject_preds)}
        pred_dict_permuted_original = {}
        for model_version in models:
            pred_dict_permuted_original[model_version] = {}
            for k in range(n_folds):
                pred_dict_permuted_original[model_version][k] = {s: pred_dict_permuted[model_version][s][k] for s in
                                                                 test_ids_list[k]}

        # Performance
        # AND: element-wise product of binary-proba -> postprocessing
        # OR: element-wise sum of binary-proba -> clip -> postprocessing
        # AVG: element-wise mean of raw-rpoba -> binary -> postprocessing
        # ¿Product of probas with 0.25/0.5 thr?

    print("Done.")

    # ensemble of model with itself
    model_version = constants.V2_TIME
    k = 0

    # single fold, single model test
    subject_ids = test_ids_list[k]
    reference_feeder_dataset = FeederDataset(dataset, subject_ids, constants.N2_RECORD, which_expert=config["expert"])
    dict_of_proba = {
        s: [
            pred_dict_original[model_version][k][s]['probability'],
            pred_dict_permuted_original[model_version][k][s]['probability']
        ] for s in subject_ids
    }
    ensemble_preds = det_utils.generate_ensemble_from_probabilities(dict_of_proba, reference_feeder_dataset)

