from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import pickle
from pprint import pprint
import shutil
import sys

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

    id_try_list = [0, 1, 2, 3]

    # ----- Prediction settings
    # Set checkpoint from where to restore, relative to results dir

    # [V11-GLOBAL-N2-MASS]
    # source_ckpt_folder = '20190619_v11_v12_global'
    # source_ckpt_subfolder = 'v11_None_None'
    # opt_thr_ss: [0.56, 0.42, 0.54, 0.44]
    # opt_thr_kc: [0.66, 0.62, 0.38, 0.64]

    # [V12-GLOBAL-N2-MASS]
    # source_ckpt_folder = '20190619_v11_v12_global'
    # source_ckpt_subfolder = 'v12_32_64'
    # opt_thr_ss: [0.6, 0.62, 0.56, 0.5]
    # opt_thr_kc: [0.46, 0.56, 0.48, 0.54]

    # [V17-GLOBAL-N2-MASS]
    # source_ckpt_folder = '20190618_grid_fb_cwtrect'
    # source_ckpt_subfolder = 'fb_0.5'
    # opt_thr_ss: [0.48, 0.6, 0.32, 0.46]
    # opt_thr_kc: [0.48, 0.58, 0.54, 0.46]

    # [V15-GLOBAL-N2-MASS]
    # source_ckpt_folder = '20190617_grid_normalization'
    # source_ckpt_subfolder = 'norm_global'
    # opt_thr_ss: [0.58, 0.42, 0.4, 0.5]
    # opt_thr_kc: [0.68, 0.58, 0.48, 0.54]

    # [V18-GLOBAL-N2-MASS]
    # source_ckpt_folder = '20190618_v18'
    # source_ckpt_subfolder = 'bsf'
    # opt_thr_ss: [0.62, 0.38, 0.4, 0.5]
    # opt_thr_kc: [0.56, 0.5, 0.56, 0.58]

    source_ckpt_folder = '20190618_v18'
    source_ckpt_subfolder = 'bsf'

    source_dataset_name = constants.MASS_KC_NAME
    target_dataset_name = constants.DREAMS_KC_NAME

    use_source_std = False

    source_task_mode = constants.N2_RECORD
    target_task_mode = constants.N2_RECORD

    verbose = True
    # -----
    tmp_dir = os.path.join(RESULTS_PATH, 'demo_predict')

    # Restore parameters
    base_path = os.path.join(
        '%s_%s_train_%s' % (
            source_ckpt_folder, source_task_mode, source_dataset_name),
        '%s' % source_ckpt_subfolder)
    ckpt_path = os.path.abspath(os.path.join(
        RESULTS_PATH, base_path, 'seed0'))
    params = pkeys.default_params.copy()
    filename = os.path.join(ckpt_path, 'params.json')
    with open(filename, 'r') as infile:
        # Overwrite previous defaults with run's params
        params.update(json.load(infile))
    print('Restoring from %s' % ckpt_path)

    # Load dataset with proper config
    target_dataset = load_dataset(target_dataset_name, params=params)
    if use_source_std:
        print('Using source dataset global std.')
        source_dataset = load_dataset(source_dataset_name, params=params)
        source_std = source_dataset.global_std
        target_dataset.global_std = source_std
        suffix = 'source_std'
    else:
        print('Using target dataset global std')
        suffix = 'target_std'

    # Perform predictions
    for k in id_try_list:
        ckpt_path = os.path.abspath(os.path.join(
            RESULTS_PATH, base_path, 'seed%d' % k))
        print('Restoring model')
        model = WaveletBLSTM(params=params, logdir=tmp_dir)
        model.load_checkpoint(os.path.join(ckpt_path, 'model', 'ckpt'))

        # Save path for predictions
        save_dir = os.path.abspath(os.path.join(
            RESULTS_PATH,
            'transfer_predictions_%s' % target_dataset_name,
            base_path, 'seed%d' % k
        ))
        checks.ensure_directory(save_dir)
        print('Predictions dir %s' % save_dir)
        print('Predicting...', flush=True)
        data_inference = FeederDataset(
            target_dataset, target_dataset.all_ids, target_task_mode)
        prediction = model.predict_dataset(data_inference, verbose=verbose)
        filename = os.path.join(
            save_dir,
            'prediction_%s_%s.pkl' % (target_task_mode, suffix))
        with open(filename, 'wb') as handle:
            pickle.dump(
                prediction,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

    # Clean tmp files
    if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    print('Done')
