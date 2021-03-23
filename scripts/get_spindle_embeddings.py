from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers import reader
from sleeprnn.common import constants
from sleeprnn.common import pkeys
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    # Config
    ckpt_folder = '20191227_bsf_10runs_e1_n2_train_mass_ss/v19'
    seeds = [0, 1, 2, 3]
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    task_mode = constants.N2_RECORD
    global_std = None
    train_fraction = 0.75
    global_thr = None

    ckpt_path = os.path.join(RESULTS_PATH, ckpt_folder)

    params = pkeys.default_params.copy()
    # update with parameters from ckpt
    # they are the same for each seed, so we simply use the first one
    params_fname = os.path.join(ckpt_path, 'seed%d' % seeds[0], 'params.json')
    with open(params_fname, 'r') as handle:
        loaded_params = json.load(handle)
    params.update(loaded_params)

    # load data
    print("Predicting on %s (expert %d)" % (dataset_name, which_expert))
    dataset = reader.load_dataset(dataset_name, params=params)
    if global_std is not None:
        dataset.global_std = global_std
        print("External global std provided. Dataset now has global std %s" % dataset.global_std)

    # load predictions
    predictions = reader.read_prediction_with_seeds(
        ckpt_folder, dataset_name, task_mode, seeds, set_list=[constants.VAL_SUBSET], parent_dataset=dataset)

    # loop through seeds
    for seed in seeds:
        print("Processing seed %d" % seed)
        # Generate split
        _, val_ids = utils.split_ids_list_v2(dataset.train_ids, split_id=seed, train_fraction=train_fraction)

        # Get signals
        x_list = dataset.get_subset_signals(
            val_ids, normalize_clip=True, normalization_mode=task_mode, which_expert=which_expert)

        # Get true spindles
        y_true_list = dataset.get_subset_stamps(
            val_ids, which_expert=which_expert, pages_subset=task_mode)
        y_true_list = [y_true.mean(axis=1).astype(np.int32) for y_true in y_true_list]

        # get predictions for this seed and set thr
        prediction_obj = predictions[seed][constants.VAL_SUBSET]
        if global_thr is None:
            used_thr = OPTIMAL_THR_FOR_CKPT_DICT[ckpt_folder][seed]
            print("Using optimized thr %s" % used_thr)
        else:
            used_thr = global_thr
            print("Using global thr %s" % used_thr)
        prediction_obj.set_probability_threshold(used_thr)
        y_pred_list = prediction_obj.get_stamps()
        y_pred_list = [y_pred.mean(axis=1).astype(np.int32) for y_pred in y_pred_list]

        # Get predicted probability at true and predicted spindles centers
        upsample_factor = 8
        probas_list = prediction_obj.get_probabilities()
        # Upsample by repetition
        probas_list = [np.stack(upsample_factor * [p], axis=1).flatten() for p in probas_list]
        y_true_probas_list = [p[y_c] for (p, y_c) in zip(probas_list, y_true_list)]
        y_pred_probas_list = [p[y_c] for (p, y_c) in zip(probas_list, y_pred_list)]

        # load checkpointed model
        logdir = os.path.join(RESULTS_PATH, 'tmp')
        model = WaveletBLSTM(params=params, logdir=logdir)
        weight_ckpt_path = os.path.join(RESULTS_PATH, ckpt_folder, 'seed%d' % seed, 'model', 'ckpt')
        print('Restoring weights from %s' % weight_ckpt_path)
        model.load_checkpoint(weight_ckpt_path)

        # Obtain tensors
        print("Generating embeddings for true spindles centers")
        embeddings_true_list = model.predict_tensor_at_samples_with_list(
            x_list, y_true_list, tensor_name='last_hidden', verbose=True)
        print("Embeddings with shape", [e.shape for e in embeddings_true_list])
        print("Generating embeddings for predicted spindles centers")
        embeddings_pred_list = model.predict_tensor_at_samples_with_list(
            x_list, y_pred_list, tensor_name='last_hidden', verbose=True)
        print("Embeddings with shape", [e.shape for e in embeddings_pred_list])

        # Organize results
        result_seed_dict = {}
        for i, subject_id in enumerate(val_ids):
            result_seed_dict[subject_id] = {
                'y_true_center': np.asarray(y_true_list[i]),
                'y_true_proba': np.asarray(y_true_probas_list[i]),
                'y_true_tensor': np.asarray(embeddings_true_list[i]),
                'y_pred_center': np.asarray(y_pred_list[i]),
                'y_pred_proba': np.asarray(y_pred_probas_list[i]),
                'y_pred_tensor': np.asarray(embeddings_pred_list[i])
            }

        # Save results
        save_path = os.path.join(RESULTS_PATH, 'embeddings', ckpt_folder, 'seed%d' % seed)
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(
            save_path,
            'embeddings_%s_thr%1.2f_%s.pkl' % (task_mode, used_thr, constants.VAL_SUBSET))
        print("Saving results at %s" % filename)
        with open(filename, 'wb') as handle:
            pickle.dump(result_seed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
