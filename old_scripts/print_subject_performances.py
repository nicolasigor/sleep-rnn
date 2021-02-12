from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pprint import pprint
import sys

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.common import constants, pkeys
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT
from sleeprnn.helpers import reader, misc

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == "__main__":
    dataset_name = constants.MASS_SS_NAME
    fs = 200
    dataset = reader.load_dataset(dataset_name, params={pkeys.FS: fs})

    # Settings
    ref_ckpt_folder = '20200724_reproduce_red_n2_train_mass_ss/v19_rep1'
    seed_id_list = [0, 1, 2, 3]

    which_expert = 1
    task_mode = constants.N2_RECORD
    set_list = [constants.VAL_SUBSET, constants.TRAIN_SUBSET]
    iou_thr = 0.2
    iou_hist_bins = np.linspace(0, 1, 21)
    iou_curve_axis = misc.custom_linspace(0.05, 0.95, 0.05)
    ids_dict = {constants.ALL_TRAIN_SUBSET: dataset.train_ids, constants.TEST_SUBSET: dataset.test_ids}
    ids_dict.update(misc.get_splits_dict(dataset, seed_id_list))

    # Load predictions
    ref_predictions_dict = reader.read_prediction_with_seeds(
        ref_ckpt_folder, dataset_name, task_mode, seed_id_list, set_list=set_list, parent_dataset=dataset)

    # Compute performance by subject
    ref_precision_dict = {}
    ref_recall_dict = {}
    for seed_id in seed_id_list:
        data_inference = FeederDataset(dataset, ids_dict[seed_id][constants.VAL_SUBSET], task_mode, which_expert)
        this_ids = data_inference.get_ids()
        this_events_list = data_inference.get_stamps()
        prediction_obj = ref_predictions_dict[seed_id][constants.VAL_SUBSET]
        prediction_obj.set_probability_threshold(OPTIMAL_THR_FOR_CKPT_DICT[ref_ckpt_folder][seed_id])
        this_detections_list = prediction_obj.get_stamps()
        for i, single_id in enumerate(this_ids):
            single_events = this_events_list[i]
            single_detections = this_detections_list[i]
            this_precision = metrics.metric_vs_iou(single_events, single_detections, [iou_thr],
                                                   metric_name=constants.PRECISION)
            this_recall = metrics.metric_vs_iou(single_events, single_detections, [iou_thr],
                                                metric_name=constants.RECALL)
            ref_precision_dict[single_id] = this_precision[0]
            ref_recall_dict[single_id] = this_recall[0]
    print("Done.")
    print("Precision Dict")
    pprint(ref_precision_dict)

    print("\nRecall Dict")
    pprint(ref_recall_dict)
