import os
import sys

PROJECT_ROOT = os.path.abspath('..')
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')


if __name__ == "__main__":

    # Load proba dataset
    byevent_proba_ckpt_path = os.path.join(
        RESULTS_PATH, 'predictions_nsrr_ss',
        'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
        'v2_time',
        'table_byevent_proba.csv'
    )

    print("Loading from checkpoint")
    table_byevent_proba = pd.read_csv(byevent_proba_ckpt_path)
    # Change 64 bits dtypes to 32 bits to save memory
    int_cols = ["female", "center_sample", "prediction_part", "category"]
    float_cols = [col for col in table_byevent_proba.columns if col not in int_cols + ["subject_id"]]
    table_byevent_proba[int_cols] = table_byevent_proba[int_cols].astype(np.int32)
    table_byevent_proba[float_cols] = table_byevent_proba[float_cols].astype(np.float32)
    print("Done.")

    # Selection of columns
    nonfeat_columns = [
        'subject_id', 'center_sample', 'prediction_part', 'category', 'probability', 'logit',
        'logit_bin', 'age', 'female']
    exclude_feat_columns = [
        'amplitude_rms',
        'c10_abs_sigma_power', 'c10_abs_sigma_power_masked',
        'c20_abs_sigma_power', 'c20_abs_sigma_power_masked',
        'c10_rel_sigma_power', 'c20_rel_sigma_power',
        'c10_density_all', 'c20_density_all',
        'c10_density_real', 'c20_density_real',
        'mean_power_0_2',
        'mean_power_2_4',
        'mean_power_4_8',
        'mean_power_8_10',
        'mean_power_16_30',
        'mean_power_4.5_30',
        'c10_rel_sigma_power_masked', 'c20_rel_sigma_power_masked',
    ]
    feat_columns = [col for col in table_byevent_proba.columns if col not in nonfeat_columns + exclude_feat_columns]
    feat_columns.append('c10_rel_sigma_power_masked')
    feat_columns.append('c20_rel_sigma_power_masked')

    # Generate context dataset
    noncontext_feats = [col for col in feat_columns if 'c10_' not in col and 'c20_' not in col]
    window_duration = 20
    fs = 200
    window_size = int(window_duration * fs)

    byevent_context_ckpt_path = os.path.join(
        RESULTS_PATH, 'predictions_nsrr_ss',
        'ckpt_20210716_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_ensemble_to_e1_n2_train_nsrr_ss',
        'v2_time',
        'table_byevent_context.csv'
    )

    subject_ids = np.unique(table_byevent_proba.subject_id)
    table_context = {'subject_id': [], 'center_sample': []}
    table_context.update({
        'c%davg_%s' % (window_duration, key): [] for key in noncontext_feats
    })
    for subject_id in subject_ids:
        subset = table_byevent_proba.loc[
            table_byevent_proba.subject_id == subject_id,
            ['center_sample', 'category'] + noncontext_feats
        ]
        event_centers = subset.center_sample.values
        subset_real = subset[subset.category == 1].drop(columns="category").set_index("center_sample")

        for event_center in event_centers:
            start_window = int(event_center - window_size // 2)
            end_window = int(start_window + window_size)
            neighbours = subset_real.loc[start_window:end_window+1]
            if event_center in neighbours.index:
                neighbours = neighbours.drop(event_center)
            # now we only have real neighbours, without considering event of interest
            if len(neighbours) > 0:
                mean_context = neighbours.mean()
            else:
                mean_context = {key: np.nan for key in noncontext_feats}
            # now save
            table_context['subject_id'].append(subject_id)
            table_context['center_sample'].append(event_center)
            for key in noncontext_feats:
                table_context['c%davg_%s' % (window_duration, key)].append(mean_context[key])
    table_context = pd.DataFrame.from_dict(table_context)

    print("Saving checkpoint")
    table_context.to_csv(byevent_context_ckpt_path, index=False)
    print("Done.")
