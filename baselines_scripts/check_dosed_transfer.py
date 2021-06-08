from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pprint import pprint
import sys

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from baselines_scripts import butils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys

BASELINE_FOLDER = '2019_chambon_dosed'
BASELINES_DATA_PATH = os.path.abspath(os.path.join(project_root, '../sleep_baselines'))

mass1_config = dict(
    dataset_name=constants.MASS_SS_NAME, which_expert=1, event_path='spindle1')
mass2_config = dict(
    dataset_name=constants.MASS_SS_NAME, which_expert=2, event_path='spindle2')
inta_config = dict(
    dataset_name=constants.INTA_SS_NAME, which_expert=1, event_path='spindle')
moda_config = dict(
    dataset_name=constants.MODA_SS_NAME, which_expert=1, event_path='spindle')


if __name__ == '__main__':
    fs = 200
    iou_threshold_report = 0.2
    average_mode = constants.MACRO_AVERAGE
    target_dataset_config = mass1_config
    source_dataset_configs = [mass2_config, moda_config, inta_config]
    normalization_mode_list = ['adapt']

    # ------------------------------------------------------

    # Fit ID
    fit_id = "%s_e%d_5cv" % (
        target_dataset_config["dataset_name"], target_dataset_config["which_expert"])
    print("\nEvaluation of %s (%s)" % (fit_id, average_mode))
    # Load dataset
    dataset = reader.load_dataset(
        target_dataset_config["dataset_name"], verbose=False, params={pkeys.FS: fs})
    # Retrieve paths
    indata_id = "from_%s_e%d_naive" % (
        target_dataset_config["dataset_name"], target_dataset_config["which_expert"])
    indata_folder = '20210605_thesis_%s_5cv_%s_final' % (
        target_dataset_config["dataset_name"].split("_")[0], target_dataset_config["event_path"])
    pred_folder_dict = {indata_id: indata_folder}
    for source_dataset_config in source_dataset_configs:
        for normalization_mode in normalization_mode_list:
            crossdata_id = "from_%s_e%d_%s" % (
                source_dataset_config["dataset_name"], source_dataset_config["which_expert"],
                normalization_mode)
            crossdata_folder = '20210607_thesis_5cv_from_%s_%s_to_%s_%s_%s' % (
                source_dataset_config["dataset_name"].split("_")[0], source_dataset_config["event_path"],
                target_dataset_config["dataset_name"].split("_")[0], target_dataset_config["event_path"],
                normalization_mode)
            pred_folder_dict[crossdata_id] = crossdata_folder
    for source_id in pred_folder_dict.keys():
        pred_dir = os.path.join(BASELINES_DATA_PATH, BASELINE_FOLDER, 'results', pred_folder_dict[source_id])
        print('\nLoading predictions from %s' % pred_dir)
        _, _, test_ids_list = butils.get_partitions(dataset, '5cv', 3)
        n_folds = len(test_ids_list)
        pred_dict = butils.get_raw_dosed_marks(pred_dir, n_folds, dataset, print_thr=False)
        pred_dict = butils.postprocess_dosed_marks(dataset, pred_dict)
        results = butils.evaluate_dosed_by_fold(
            dataset, target_dataset_config["which_expert"], test_ids_list, pred_dict, average_mode, iou_threshold_report)
        print("Test performance (%s, iou >= %1.2f, from train to test), %s to %s" % (
            average_mode, iou_threshold_report, source_id, fit_id))
        butils.print_performance(results)

