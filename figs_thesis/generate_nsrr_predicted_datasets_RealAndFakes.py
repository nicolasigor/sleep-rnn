import os
import sys
import pickle
import time

PROJECT_ROOT = os.path.abspath('..')
sys.path.append(PROJECT_ROOT)

import numpy as np

from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants, pkeys
from figs_thesis import fig_utils

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')


if __name__ == "__main__":
    nsrr_preds = fig_utils.PredictedNSRR()
    nsrr = load_dataset(constants.NSRR_SS_NAME, load_checkpoint=True, params={pkeys.PAGE_DURATION: 30})

    save_dir = os.path.abspath(os.path.join(
        RESULTS_PATH,
        'predictions_nsrr_ss',
        'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
        'v2_time'
    ))
    os.makedirs(save_dir, exist_ok=True)

    n_folds = 116
    fold_ids = np.arange(n_folds)

    size_of_parts = 10
    n_parts = int(np.ceil(n_folds / size_of_parts))

    start_time = time.time()
    for part in range(n_parts):
        start_fold = part * size_of_parts
        end_fold = (part + 1) * size_of_parts
        part_folds = fold_ids[start_fold:end_fold]

        # Real
        predictions = nsrr_preds.get_predictions(part_folds, nsrr)
        predictions.delete_parent_dataset()

        file_path = os.path.join(save_dir, 'prediction_part%d.pkl' % part)
        with open(file_path, 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Fake
        # Lower threshold to allow fake spindles
        predictions = nsrr_preds.get_predictions(part_folds, nsrr, threshold=0.25)
        predictions.delete_parent_dataset()

        file_path = os.path.join(save_dir, 'prediction_0.25_part%d.pkl' % part)
        with open(file_path, 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elap_time = time.time() - start_time
        print("[E.T. %1.1f s] Part %d saved at %s" % (elap_time, part, file_path))
