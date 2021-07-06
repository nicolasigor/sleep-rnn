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
from figs_thesis import fig_utils

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == "__main__":

    fname = 'cheating_thresholds.csv'

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
    thr_search_space = np.arange(0.1, 0.9 + 0.001, 0.02)
    thr_search_space = np.round(thr_search_space, 2)

    table = {
        'dataset': [],
        'model': [],
        'fold': [],
        'subject': [],
        'cheat_thr': [],
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
                    print("Starting subject %s (E.T. %1.4f s)" % (subject_id, time.time() - start_time))

                    # Get events
                    events = dataset.get_subject_stamps(
                        subject_id, which_expert=config['expert'], pages_subset=constants.N2_RECORD)
                    # Prepare model predictions
                    detections_at_thr_list = []
                    for thr in thr_search_space:
                        tmp_dict[k][constants.TEST_SUBSET].set_probability_threshold(thr)
                        detections = tmp_dict[k][constants.TEST_SUBSET].get_subject_stamps(subject_id)
                        detections_at_thr_list.append(detections)
                    # Compute performance
                    af1_list = Parallel(n_jobs=-1)(
                        delayed(metrics.average_metric_macro_average)([events], [detections])
                        for detections in detections_at_thr_list
                    )
                    # Get thr that generates max AF1 at this subject
                    max_idx = np.argmax(af1_list).item()
                    best_thr = thr_search_space[max_idx]

                    # Save in global table
                    table['dataset'].append(dataset_str)
                    table['model'].append(model_version)
                    table['fold'].append(k)
                    table['subject'].append(str(subject_id))
                    table['cheat_thr'].append(best_thr)

    # Transform to pandas and save
    table = pd.DataFrame.from_dict(table)
    table.to_csv(fname, index=False, float_format='%1.2f')
