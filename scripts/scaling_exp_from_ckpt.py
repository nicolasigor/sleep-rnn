from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
from pprint import pprint
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.data.loader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import pkeys
from sleeprnn.common import checks

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':

    id_try = 0
    scale_list = 0.5 + np.arange(20+1) / 20
    scale_list = np.round(scale_list, decimals=2).astype(np.float32)
    print('Scaling experiment with scales:', scale_list)

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir
    ckpt_folder = os.path.join(
        '20190617_grid_normalization_n2_train_mass_ss',
        'norm_global')
    task_mode = constants.N2_RECORD
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    verbose = True
    grid_folder_list = None
    # -----

    print('\nModel predicting on %s_%s' % (dataset_name, task_mode))
    ckpt_path = os.path.abspath(os.path.join(
        RESULTS_PATH,
        ckpt_folder,
        'seed%d' % id_try
    ))
    # Restore params of ckpt
    params = pkeys.default_params.copy()
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        # Overwrite previous defaults with run's params
        params.update(json.load(infile))
    print('Restoring from %s' % ckpt_path)

    # Create model
    print('Restoring model')
    model = WaveletBLSTM(
        params,
        logdir=os.path.join(RESULTS_PATH, 'demo_predict'))
    # Load checkpoint
    model.load_checkpoint(
        os.path.join(ckpt_path, 'model', 'ckpt'))
    print('Model restored.')

    # Save path for predictions
    save_dir = os.path.abspath(os.path.join(
        RESULTS_PATH,
        'scaling_results',
        ckpt_folder,
        'seed%d' % id_try
    ))
    checks.ensure_directory(save_dir)

    dataset = load_dataset(dataset_name, params=params)  # Create dataset

    # pprint(params)

    # Predict
    print('Predictions will be saved at %s' % save_dir)
    print('Predicting test set', flush=True)
    test_ids = dataset.test_ids  # Test data
    data_inference = FeederDataset(
        dataset, test_ids, task_mode,
        which_expert=which_expert)
    for this_scale in scale_list:
        print('Evaluating scale %1.2f' % this_scale, flush=True)
        prediction = model.predict_dataset(
            data_inference, verbose=verbose, input_scale_factor=this_scale)
        filename = os.path.join(
            save_dir,
            'prediction_%s_test_s%1.2f.pkl' % (task_mode, this_scale))
        print('Prediction saved at %s' % filename)
        with open(filename, 'wb') as handle:
            pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Predict
    print('Predicting train set', flush=True)
    train_ids = dataset.train_ids  # Alltrain data
    data_inference = FeederDataset(
        dataset, train_ids, task_mode,
        which_expert=which_expert)
    for this_scale in scale_list:
        print('Evaluating scale %1.2f' % this_scale, flush=True)
        prediction = model.predict_dataset(
            data_inference, verbose=verbose, input_scale_factor=this_scale)
        filename = os.path.join(
            save_dir,
            'prediction_%s_train_s%1.2f.pkl' % (task_mode, this_scale))
        print('Prediction saved at %s' % filename)
        with open(filename, 'wb') as handle:
            pickle.dump(prediction, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)




