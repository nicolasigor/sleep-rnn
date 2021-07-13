import os
import sys
from joblib import delayed, Parallel
import time

import numpy as np
import pandas as pd

project_root = '..'
sys.path.append(project_root)

from sleeprnn.detection import metrics
from sleeprnn.common import constants
from sleeprnn.helpers import reader
from sleeprnn.data import utils
from figs_thesis import fig_utils

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == "__main__":

    fname = 'adapted_thresholds.csv'

    models = [constants.V2_TIME, constants.V2_CWT1D]
    print_dataset_names = {
        (constants.MASS_SS_NAME, 1): "MASS-SS2-E1SS",
        (constants.MASS_SS_NAME, 2): "MASS-SS2-E2SS",
        (constants.MODA_SS_NAME, 1): "MASS-MODA",
        (constants.INTA_SS_NAME, 1): "INTA-UCH",
    }
    eval_configs = [
        dict(dataset_name=constants.MASS_SS_NAME, expert=1, strategy='5cv', seeds=3),
        dict(dataset_name=constants.MASS_SS_NAME, expert=2, strategy='5cv', seeds=3),
        dict(dataset_name=constants.MODA_SS_NAME, expert=1, strategy='5cv', seeds=3),
        dict(dataset_name=constants.INTA_SS_NAME, expert=1, strategy='5cv', seeds=3),
    ]
    thr_search_space = np.arange(0.04, 0.96 + 0.001, 0.02)
    thr_search_space = np.round(thr_search_space, 2)

    minutes_list = [3, 5, 7.5, 10, 15, 20, 25, 30]
    min_events_to_fit = 2

    table = {
        'dataset': [],
        'model': [],
        'fold': [],
        'subject': [],
        'minutes': [],
        'adapt_thr': [],
    }

    start_time = time.time()
    for config in eval_configs:
        print("\nLoading", config)
        dataset_str = print_dataset_names[(config['dataset_name'], config['expert'])]
        print("Pretty name: %s" % dataset_str)
        dataset = reader.load_dataset(config["dataset_name"], verbose=False)

        # In MODA, only some subjects are used (N=28)
        if config["dataset_name"] == constants.MODA_SS_NAME:
            valid_subjects = [
                sub_id for sub_id in dataset.all_ids
                if (dataset.data[sub_id]['n_blocks'] == 10)
                   and (sub_id not in ['01-01-0012', '01-01-0022'])
            ]
            print("moda, using n=", len(valid_subjects))
        else:
            valid_subjects = dataset.all_ids

        for model_version in models:
            print("\nProcessing model %s" % model_version)
            tmp_dict = fig_utils.get_red_predictions(
                model_version, config["strategy"], dataset, config["expert"], verbose=False)
            for k in tmp_dict.keys():
                print("\nFold %d" % k)
                fold_subjects = tmp_dict[k][constants.TEST_SUBSET].all_ids
                for subject_id in fold_subjects:
                    if subject_id not in valid_subjects:
                        continue
                    print("Starting subject %s" % subject_id)

                    n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
                    # Get events
                    events = dataset.get_subject_stamps(
                        subject_id, which_expert=config['expert'], pages_subset=constants.N2_RECORD)
                    # Prepare model predictions
                    detections_at_thr_list = []
                    for thr in thr_search_space:
                        tmp_dict[k][constants.TEST_SUBSET].set_probability_threshold(thr)
                        detections = tmp_dict[k][constants.TEST_SUBSET].get_subject_stamps(subject_id)
                        detections_at_thr_list.append(detections)
                    # Loop through minutes
                    for minutes in minutes_list:
                        n_first_pages = int(minutes * 60 / dataset.page_duration)
                        first_n2_pages = n2_pages[:n_first_pages]
                        print("Using %1.1f minutes (the first %d N2 pages)" % (minutes, n_first_pages))
                        # Filter events
                        first_events = utils.extract_pages_for_stamps(events, first_n2_pages, dataset.page_size)
                        if first_events.shape[0] < min_events_to_fit:
                            print("Too few events (%d). Skipped" % (first_events.shape[0]))
                        # Filter detections
                        first_detections_at_thr_list = [
                            utils.extract_pages_for_stamps(dets, first_n2_pages, dataset.page_size)
                            for dets in detections_at_thr_list
                        ]

                        # Compute performance
                        af1_list = Parallel(n_jobs=-1)(
                            delayed(metrics.average_metric_macro_average)([first_events], [first_detections])
                            for first_detections in first_detections_at_thr_list
                        )

                        # Get thr that generates max AF1 at this subject
                        max_idx = np.argmax(af1_list).item()
                        best_thr = thr_search_space[max_idx]
                        best_af1 = af1_list[max_idx]
                        print("Best AF1 %1.1f at thr %1.2f (E.T. %1.4f s)" % (100 * best_af1, best_thr, time.time() - start_time))

                        # Save in global table
                        table['dataset'].append(dataset_str)
                        table['model'].append(model_version)
                        table['fold'].append(k)
                        table['subject'].append(str(subject_id))
                        table['minutes'].append(float(minutes))
                        table['adapt_thr'].append(best_thr)

    # Transform to pandas and save
    table = pd.DataFrame.from_dict(table)
    table.to_csv(fname, index=False, float_format='%1.2f')
