import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

PATH_THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(PATH_THIS_DIR, '..'))
BASELINES_PATH = os.path.join(PROJECT_ROOT, 'resources', 'comparison_data', 'baselines_2021')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

from sleeprnn.data import utils
from sleeprnn.helpers import reader
from sleeprnn.common import constants, viz
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT
from sleeprnn.detection import metrics


def get_baseline_predictions(
        baseline_name,
        strategy,
        target_dataset_name,
        target_expert,
        source_dataset_name=None,
        source_expert=None,
):
    baseline_folders = {
        'spinky': '2017_lajnef_spinky',
        'a7': '2019_lacourse_a7',
        'dosed': '2019_chambon_dosed'
    }
    baseline_folder = baseline_folders[baseline_name]

    target_string = '%s_e%d_%s' % (target_dataset_name, target_expert, strategy)
    if source_dataset_name is not None:
        source_string = '%s_e%d_%s' % (source_dataset_name, source_expert, strategy)
    else:
        source_string = target_string
    filename = '%s_from_%s.pkl' % (target_string, source_string)
    filepath = os.path.join(BASELINES_PATH, baseline_folder, filename)
    with open(filepath, 'rb') as handle:
        predictions_dict = pickle.load(handle)
    # output is predictions_dict[fold_id][subject_id]
    return predictions_dict


def get_red_predictions_for_perturbations(
        model_version,
        strategy,
        dataset,
        expert,
        task_mode=constants.N2_RECORD,
        ckpt_folder_prefix='20210529_thesis_indata',
        perturbation_date='20210620',
        verbose=False
):
    ckpt_folder = '%s_from_%s_desc_perturbation_to_%s' % (
        perturbation_date,
        '%s_%s_e%d_%s_train_%s' % (
            ckpt_folder_prefix, strategy, expert, task_mode, dataset.dataset_name),
        'e%d_%s_train_%s' % (expert, task_mode, dataset.dataset_name)
    )
    available_grid_folders = os.listdir(os.path.join(
        RESULTS_PATH, 'predictions_%s' % dataset.dataset_name, ckpt_folder))
    available_grid_folders.sort()
    available_grid_folders = [n for n in available_grid_folders if model_version in n]
    perturbation_predictions_dict = {}
    for grid_folder in available_grid_folders:
        perturbation_id = grid_folder.split("_")[-1]
        grid_folder_complete = os.path.join(ckpt_folder, grid_folder)
        print("Loading predictions from %s" % grid_folder_complete) if verbose else None
        predictions_dict = reader.read_predictions_crossval(grid_folder_complete, dataset, task_mode)
        # Ensure optimal threshold in predictions
        opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[grid_folder_complete]
        for k in predictions_dict.keys():
            for set_name in predictions_dict[k].keys():
                predictions_dict[k][set_name].set_probability_threshold(opt_thr_list[k])
        # output is predictions_dict[fold_id][subset] -> [subject_id]
        perturbation_predictions_dict[perturbation_id] = predictions_dict
    return perturbation_predictions_dict


def get_red_predictions_for_pink(
        model_version,
        strategy,
        source_dataset,
        source_expert,
        pink_dataset=None,
        task_mode=constants.N2_RECORD,
        ckpt_folder_prefix='20210529_thesis_indata',
        pink_date='20210621',
        verbose=False
):
    if pink_dataset is None:
        pink_dataset = reader.load_dataset(constants.PINK_NAME, verbose=False)
        pink_dataset.event_name = source_dataset.event_name
    else:
        if pink_dataset.event_name != source_dataset.event_name:
            raise ValueError("Provided PINK dataset has event %s but source has event %s" % (
                pink_dataset.event_name, source_dataset.event_name
            ))
    ckpt_folder = '%s_from_%s_desc_pink_to_%s' % (
        pink_date,
        '%s_%s_e%d_%s_train_%s' % (
            ckpt_folder_prefix, strategy, source_expert, task_mode, source_dataset.dataset_name),
        'e%d_%s_train_%s' % (1, task_mode, pink_dataset.dataset_name)
    )
    available_grid_folders = os.listdir(os.path.join(
        RESULTS_PATH, 'predictions_%s' % pink_dataset.dataset_name, ckpt_folder))
    available_grid_folders.sort()
    available_grid_folders = [n for n in available_grid_folders if model_version in n]
    pink_predictions_dict = {}
    for grid_folder in available_grid_folders:
        perturbation_id = grid_folder.split("_")[-1]
        grid_folder_complete = os.path.join(ckpt_folder, grid_folder)
        print("Loading predictions from %s" % grid_folder_complete) if verbose else None
        predictions_dict = reader.read_predictions_crossval(grid_folder_complete, pink_dataset, task_mode)
        # Ensure optimal threshold in predictions
        opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[grid_folder_complete]
        for k in predictions_dict.keys():
            for set_name in predictions_dict[k].keys():
                predictions_dict[k][set_name].set_probability_threshold(opt_thr_list[k])
        # output is predictions_dict[fold_id][subset] -> [subject_id]
        pink_predictions_dict[perturbation_id] = predictions_dict
    return pink_predictions_dict


def get_red_predictions(
        model_version,
        strategy,
        target_dataset,
        target_expert,
        source_dataset_name=None,
        source_expert=None,
        task_mode=constants.N2_RECORD,
        ckpt_folder_prefix='20210529_thesis_indata',
        transfer_date='20210605',
        transfer_desc='sourcestd',
        verbose=False
):
    # Form ckpt_folder
    if source_dataset_name is None:
        source_dataset_name = target_dataset.dataset_name
        source_expert = target_expert

    if (source_dataset_name == target_dataset.dataset_name) and (source_expert == target_expert):
        # In-dataset
        ckpt_folder = '%s_%s_e%d_%s_train_%s' % (
            ckpt_folder_prefix, strategy, target_expert, task_mode, target_dataset.dataset_name
        )
    else:
        # Cross-dataset
        ckpt_folder = '%s_from_%s_desc_%s_to_%s' % (
            transfer_date,
            '%s_%s_e%d_%s_train_%s' % (
                ckpt_folder_prefix, strategy, source_expert, task_mode, source_dataset_name),
            transfer_desc,
            'e%d_%s_train_%s' % (target_expert, task_mode, target_dataset.dataset_name)
        )
    grid_folder_complete = os.path.join(ckpt_folder, model_version)
    print("Loading predictions from %s" % grid_folder_complete) if verbose else None
    predictions_dict = reader.read_predictions_crossval(grid_folder_complete, target_dataset, task_mode)
    # Ensure optimal threshold in predictions
    opt_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[grid_folder_complete]
    for k in predictions_dict.keys():
        for set_name in predictions_dict[k].keys():
            predictions_dict[k][set_name].set_probability_threshold(opt_thr_list[k])
    # output is predictions_dict[fold_id][subset] -> [subject_id]
    return predictions_dict


def compute_fold_performance_vs_iou(events_list, detections_list, average_mode, iou_axis):
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average}
    # Compute performance
    iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
    f1_score = metric_vs_iou_fn_dict[average_mode](
        events_list, detections_list, iou_axis,
        iou_matching_list=iou_matching_list, metric_name=constants.F1_SCORE)
    recall = metric_vs_iou_fn_dict[average_mode](
        events_list, detections_list, iou_axis,
        iou_matching_list=iou_matching_list, metric_name=constants.RECALL)
    precision = metric_vs_iou_fn_dict[average_mode](
        events_list, detections_list, iou_axis,
        iou_matching_list=iou_matching_list, metric_name=constants.PRECISION)
    nonzero_iou_list = [iou_matching[iou_matching > 0] for iou_matching in iou_matching_list]
    outputs = {
        'F1-score_vs_iou': f1_score,
        'Recall_vs_iou': recall,
        'Precision_vs_iou': precision,
        'nonzero_IoU': nonzero_iou_list
    }
    return outputs


def compute_miou(nonzero_iou_list, average_mode):
    if average_mode == constants.MACRO_AVERAGE:
        miou_list = [np.mean(nonzero_iou) for nonzero_iou in nonzero_iou_list]
        miou = np.mean(miou_list)
    elif average_mode == constants.MICRO_AVERAGE:
        miou = np.concatenate(nonzero_iou_list).mean()
    else:
        raise ValueError("Average mode %s invalid" % average_mode)
    return miou


def compute_fold_performance(events_list, detections_list, average_mode, iou_threshold_report=0.2):
    # Compute performance
    outputs_vs_iou = compute_fold_performance_vs_iou(
        events_list, detections_list, average_mode, [iou_threshold_report])
    miou = compute_miou(outputs_vs_iou['nonzero_IoU'], average_mode)
    outputs = {
        'F1-score': outputs_vs_iou['F1-score_vs_iou'][0],
        'Recall': outputs_vs_iou['Recall_vs_iou'][0],
        'Precision': outputs_vs_iou['Precision_vs_iou'][0],
        'mIoU': miou
    }
    return outputs


def compute_subject_performance(events_list, detections_list, iou_threshold_report=0.2):
    # Compute performance
    iou_matching_list, _ = metrics.matching_with_list(events_list, detections_list)
    f1_score = metrics.metric_vs_iou_macro_average(
        events_list, detections_list, [iou_threshold_report],
        iou_matching_list=iou_matching_list, metric_name=constants.F1_SCORE, collapse_values=False)
    recall = metrics.metric_vs_iou_macro_average(
        events_list, detections_list, [iou_threshold_report],
        iou_matching_list=iou_matching_list, metric_name=constants.RECALL, collapse_values=False)
    precision = metrics.metric_vs_iou_macro_average(
        events_list, detections_list, [iou_threshold_report],
        iou_matching_list=iou_matching_list, metric_name=constants.PRECISION, collapse_values=False)
    nonzero_iou_list = [m[m > 0] for m in iou_matching_list]
    miou_list = []
    for nonzero_iou in nonzero_iou_list:
        if nonzero_iou.size > 0:
            miou_list.append(np.mean(nonzero_iou))
        else:
            miou_list.append(np.nan)
    miou_list = np.array(miou_list)
    outputs = {
        'F1-score': f1_score[:, 0],
        'Recall': recall[:, 0],
        'Precision': precision[:, 0],
        'mIoU': miou_list
    }
    return outputs


def format_metric(mean_value, std_value, scale_by=100):
    return "$%1.1f\pm %1.1f$" % (scale_by * mean_value, scale_by * std_value)


def linear_regression(durations_x, durations_y, min_dur, max_dur, ax, **kwargs):
    durations_x = durations_x.reshape(-1, 1)
    reg = LinearRegression().fit(durations_x, durations_y)
    # reg = HuberRegressor().fit(durations_x, durations_y)
    r2_score = reg.score(durations_x, durations_y)
    x_reg = np.array([min_dur, max_dur]).reshape(-1, 1)
    y_reg = reg.predict(x_reg)
    ax.plot(
        x_reg, y_reg, linestyle='--', zorder=20,
        color=viz.PALETTE['red'], linewidth=0.8, label='$R^2$ = %1.2f' % r2_score)
    ax.legend(**kwargs)
    return r2_score


def compute_iou_histogram(nonzero_iou_subject_list, average_mode, iou_hist_bins):
    if average_mode == constants.MICRO_AVERAGE:
        nonzero_iou_subject_list = [np.concatenate(nonzero_iou_subject_list)]
    iou_mean = []
    iou_hist = []
    for nz_iou in nonzero_iou_subject_list:
        iou_mean.append(np.mean(nz_iou))
        this_iou_hist, _ = np.histogram(nz_iou, bins=iou_hist_bins, density=True)
        iou_hist.append(this_iou_hist)
    iou_mean = np.mean(iou_mean)
    iou_hist_values = np.stack(iou_hist, axis=0).mean(axis=0)
    return iou_mean, iou_hist_values


# def get_frequency_by_peaks(x, fs, distance_in_seconds=0.04, use_median=False):
#     distance = int(fs * distance_in_seconds)
#     peaks_max, _ = find_peaks(x, distance=distance)
#     peaks_min, _ = find_peaks(-x, distance=distance)
#     dist_maxmax = np.diff(peaks_max)
#     dist_minmin = np.diff(peaks_min)
#     dist_pp = np.concatenate([dist_maxmax, dist_minmin])
#
#     # Filter
#     min_val, max_val = np.percentile(dist_pp, (10, 90))
#     dist_pp = dist_pp[(dist_pp >= min_val) & (dist_pp <= max_val)]
#
#     # Reduce
#     if use_median:
#         dist_pp = np.percentile(dist_pp, 50, interpolation='linear')
#     else:
#         dist_pp = np.mean(dist_pp)
#
#     freq_count = fs / dist_pp
#     return freq_count


# def get_frequency_by_zero_crossings(
#         x, fs,
#         distance_in_seconds=0.02,
#         upsampling_factor=100,
#         use_median=True
# ):
#     t = np.arange(x.size) / fs
#     fs_new = upsampling_factor * fs
#     distance = int(fs_new * distance_in_seconds)
#     # Densify
#     t_new = np.arange(t[0], t[-1], 1.0 / fs_new)
#     x_new = interp1d(t, x)(t_new)
#     # Find crossings
#     pos = x_new > 0
#     npos = ~pos
#     zero_crossings = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
#     inter_crossing_samples = np.diff(zero_crossings)
#     # Filter crossing too close
#     inter_crossing_samples = inter_crossing_samples[inter_crossing_samples > distance]
#     # Reduce
#     if use_median:
#         avg_inter_crossing_samples = np.percentile(inter_crossing_samples, 50, interpolation='linear')
#     else:
#         avg_inter_crossing_samples = np.mean(inter_crossing_samples)
#     # Estimate
#     freq_estimated = fs_new / (2.0 * avg_inter_crossing_samples)
#     return freq_estimated


def get_fft_spectrum(x, fs, pad_to_duration=10, f_min=1, f_max=30, apply_hann_window=False):
    x_base = np.zeros(fs * pad_to_duration)
    if apply_hann_window:
        x = x * np.hanning(x.size)
    start_sample = (x_base.size - x.size) // 2
    end_sample = start_sample + x.size
    x_base[start_sample:end_sample] = x
    x = x_base
    y = np.fft.rfft(x) / x.size
    y = np.abs(y)
    f = np.fft.rfftfreq(x.size, d=1. / fs)
    y = y[(f >= f_min) & (f <= f_max)]
    f = f[(f >= f_min) & (f <= f_max)]
    return f, y


def get_frequency_by_fft(x, fs, pad_to_duration=10, f_min=5, f_max=30, apply_hann_window=False):
    f, y = get_fft_spectrum(
        x, fs, pad_to_duration=pad_to_duration, f_min=f_min, f_max=f_max, apply_hann_window=apply_hann_window)
    freq_fft = f[np.argmax(y)]
    return freq_fft


def get_amplitude_spindle(x, fs, distance_in_seconds=0.04):
    distance = int(fs * distance_in_seconds)
    peaks_max, _ = find_peaks(x, distance=distance)
    peaks_min, _ = find_peaks(-x, distance=distance)
    peaks = np.sort(np.concatenate([peaks_max, peaks_min]))
    peak_values = x[peaks]
    peak_to_peak_diff = np.abs(np.diff(peak_values))
    max_pp = np.max(peak_to_peak_diff)
    return max_pp


def find_kcomplex_negative_peak(x, fs, left_edge_tol=0.05, right_edge_tol=0.1):
    # Find negative peak
    left_edge_tol = int(fs * left_edge_tol)
    right_edge_tol = int(fs * right_edge_tol)
    negative_peaks, _ = find_peaks(-x)
    negative_peaks = [
        peak for peak in negative_peaks
        if left_edge_tol < peak < x.size - right_edge_tol]
    negative_peaks = np.array(negative_peaks, dtype=np.int32)
    negative_peaks_values = x[negative_peaks]
    negative_peak_loc = negative_peaks[np.argmin(negative_peaks_values)]
    return negative_peak_loc


def get_amplitude_kcomplex(x, fs):
    amplitude = x.max() - x.min()
    return amplitude


def get_amplitude_event(x, fs, event_name):
    if event_name == 'spindle':
        return get_amplitude_spindle(x, fs)
    elif event_name == 'kcomplex':
        return get_amplitude_kcomplex(x, fs)
    else:
        raise ValueError()


def get_filtered_signal_for_event(x, fs, event_name):
    if event_name == 'spindle':
        return utils.apply_bandpass(x, fs, lowcut=9.5, highcut=16.5)
    elif event_name == 'kcomplex':
        return utils.apply_lowpass(x, fs, cutoff=8)
    else:
        raise ValueError()
