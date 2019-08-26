from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
from pprint import pprint
import sys
import itertools

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import pkeys
from sleeprnn.common import checks

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    save_dir = os.path.join(RESULTS_PATH, 'lrp_dataset')
    checks.ensure_directory(save_dir)

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir
    lrp_id = '20190713_v11_64_128_256'
    ckpt_folder = os.path.join('20190713_report_v11_n2_train_mass_ss', '64_128_256')
    which_seed = 1
    optimal_thr = 0.64
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    verbose = True
    # -----
    ckpt_path = os.path.join(
        RESULTS_PATH, ckpt_folder, 'seed%d' % which_seed)

    print('\nModel predicting on %s_%s' % (dataset_name, task_mode))
    dataset = load_dataset(dataset_name)
    test_ids = dataset.test_ids
    data_inference = FeederDataset(
        dataset, test_ids, task_mode,
        which_expert=which_expert)

    # Restore params of ckpt
    params = pkeys.default_params.copy()
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        # Overwrite previous defaults with run's params
        params.update(json.load(infile))
    print('Restoring from %s' % ckpt_path)

    # Obtain data
    print('Getting input and output data')
    border_size = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
    x, y = data_inference.get_data(
        border_size=border_size, pages_subset=task_mode,
        normalization_mode=task_mode, verbose=verbose)
    pages = data_inference.get_pages(pages_subset=task_mode, verbose=verbose)

    # Labels are downsampled by 8 and without border
    down_factor = params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
    y = [this_y[:, border_size:-border_size:down_factor] for this_y in y]
    stamps = data_inference.get_stamps()

    print('x', [this_data.shape for this_data in x])
    print('y', [this_data.shape for this_data in y])
    print('pages', [this_data.shape for this_data in pages])
    print('stamps', [this_data.shape for this_data in stamps])

    print('Ready')

    # Create model
    print('Restoring model')
    params[pkeys.PREDICT_WITH_AUGMENTED_PAGE] = False
    model = WaveletBLSTM(
        params, logdir=os.path.join(RESULTS_PATH, 'demo_predict'))
    model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

    # # Get spectrograms
    # print('Computing CWT before batchnorm')
    # cwt = []
    # for single_x in x:
    #     single_cwt = model.compute_cwt(single_x)
    #     cwt.append(single_cwt)
    # print('cwt', [this_data.shape for this_data in cwt])

    # Predict
    print('Predicting test set', flush=True)
    prediction = model.predict_dataset(
        data_inference, verbose=verbose)
    # Set optimal thr
    prediction.set_probability_threshold(optimal_thr)
    predicted_stamps = prediction.get_stamps()
    print('predicted_stamps', [this_data.shape for this_data in predicted_stamps])

    predicted_y = prediction.get_probabilities()
    page_size_down = int(
        params[pkeys.PAGE_DURATION] * params[pkeys.FS] / down_factor)
    predicted_y = [
        utils.extract_pages(
            single_y_hat, single_pages, page_size_down, border_size=0)
        for (single_y_hat, single_pages) in zip(predicted_y, pages)
    ]
    print('predicted_y', [this_data.shape for this_data in predicted_y])

    # Saving
    for k, sub_id in enumerate(data_inference.get_ids()):
        filename = os.path.join(save_dir, '%s_data_s%02d' % (lrp_id, sub_id))
        np.savez(
            filename,
            x=x[k].astype(np.float32),
            y=y[k].astype(np.int32),
            stamps=stamps[k].astype(np.int32),
            pages=pages[k].astype(np.int16),
            # cwt=cwt[k].astype(np.float32),
            predicted_stamps=predicted_stamps[k].astype(np.int32),
            predicted_y=predicted_y[k].astype(np.float16),
            optimal_thr=optimal_thr,
            seed=which_seed,
            ckpt_path=ckpt_path
        )
        print('%s saved.' % filename, flush=True)
