import numpy as np


reports = [
    'AF1 66.77/1.23 [0.5] 66.69/1.59 [0.50, 0.46, 0.48, 0.42, 0.54], F 79.43/1.17, P 78.6/ 9.9, R 82.8/10.9 for v43_cv2_k3l2_linear_resid',
    'AF1 66.92/1.80 [0.5] 66.65/2.29 [0.50, 0.50, 0.46, 0.38, 0.50], F 79.60/1.39, P 79.0/ 9.6, R 82.4/10.3 for v43_cv2_k3l2_none___resid',
    'AF1 66.83/4.12 [0.5] 66.61/4.07 [0.44, 0.40, 0.50, 0.50, 0.54], F 79.69/3.64, P 77.4/ 9.1, R 84.0/ 8.8 for v43_cv0_k3l2_nonlin_resid',
    'AF1 66.56/1.39 [0.5] 66.49/1.61 [0.48, 0.50, 0.42, 0.44, 0.50], F 79.40/1.16, P 78.5/ 9.5, R 82.7/10.9 for v43_cv2_k7l1_none___plain',
    'AF1 66.27/1.28 [0.5] 66.49/1.71 [0.48, 0.44, 0.46, 0.40, 0.58], F 79.29/1.40, P 79.2/10.3, R 82.1/11.5 for v43_cv2_k3l2_nonlin_resid',
    'AF1 66.19/4.28 [0.5] 66.46/4.12 [0.44, 0.52, 0.44, 0.48, 0.60], F 79.65/3.36, P 79.2/ 9.8, R 82.1/ 8.9 for v43_cv0_k7l1_none___plain',
    'AF1 66.48/1.73 [0.5] 66.44/1.88 [0.52, 0.40, 0.40, 0.42, 0.54], F 79.58/1.13, P 78.6/ 9.2, R 82.7/ 9.8 for v43_cv2_k3l2_nonlin_plain',
    'AF1 66.14/1.56 [0.5] 66.40/1.66 [0.46, 0.42, 0.44, 0.46, 0.50], F 79.24/1.63, P 79.8/ 8.5, R 80.8/11.3 for v43_cv2_k3l2_none___plain',
    'AF1 66.35/3.96 [0.5] 66.37/4.03 [0.52, 0.52, 0.50, 0.50, 0.50], F 79.73/3.45, P 78.4/ 9.4, R 83.0/ 8.4 for v43_cv0_k3l2_none___resid',
    'AF1 66.45/1.82 [0.5] 66.29/2.07 [0.50, 0.50, 0.38, 0.38, 0.52], F 79.22/1.50, P 78.1/10.1, R 82.9/10.7 for v43_cv2_k7l1_none___resid',
    'AF1 66.06/4.48 [0.5] 66.24/4.39 [0.44, 0.48, 0.52, 0.48, 0.50], F 79.29/4.00, P 78.8/ 9.7, R 81.8/ 9.2 for v43_cv0_k3l2_none___plain',
    'AF1 66.13/4.23 [0.5] 66.18/4.05 [0.50, 0.44, 0.48, 0.54, 0.54], F 79.49/3.51, P 78.0/ 9.8, R 83.2/ 9.3 for v43_cv0_k3l2_linear_resid',
    'AF1 66.17/4.28 [0.5] 66.05/4.28 [0.48, 0.44, 0.46, 0.48, 0.54], F 79.18/3.77, P 79.1/ 9.2, R 81.5/10.7 for v43_cv0_k7l1_none___resid',
    'AF1 65.85/4.50 [0.5] 65.97/4.44 [0.46, 0.50, 0.46, 0.50, 0.48], F 79.14/3.75, P 78.5/ 9.9, R 82.2/10.4 for v43_cv0_k7l1_linear_resid',
    'AF1 65.79/2.26 [0.5] 65.89/1.82 [0.56, 0.50, 0.26, 0.34, 0.48], F 79.08/1.57, P 78.1/ 9.7, R 82.3/10.4 for v43_cv2_k7l1_nonlin_plain',
    'AF1 65.62/4.54 [0.5] 65.55/4.45 [0.50, 0.48, 0.50, 0.50, 0.50], F 79.05/3.84, P 77.7/ 9.9, R 82.7/10.0 for v43_cv0_k7l1_nonlin_plain',
    'AF1 65.54/3.95 [0.5] 65.54/4.00 [0.52, 0.52, 0.52, 0.50, 0.48], F 79.11/3.20, P 76.2/ 9.1, R 84.3/ 9.4 for v43_cv0_k7l1_nonlin_resid',
    'AF1 65.44/1.91 [0.5] 65.52/1.77 [0.46, 0.46, 0.50, 0.44, 0.50], F 78.62/2.03, P 79.5/ 9.6, R 80.5/12.6 for v43_cv2_k7l1_linear_resid',
    'AF1 65.33/3.77 [0.5] 65.51/3.81 [0.50, 0.56, 0.56, 0.54, 0.48], F 78.82/3.33, P 77.2/10.1, R 82.7/ 9.0 for v43_cv0_k3l2_nonlin_plain',
    'AF1 65.24/1.70 [0.5] 65.43/1.41 [0.28, 0.54, 0.54, 0.50, 0.46], F 78.36/1.48, P 77.6/ 9.7, R 81.8/11.3 for v43_cv1_k3l2_nonlin_plain',
    'AF1 65.03/2.09 [0.5] 65.28/2.05 [0.48, 0.52, 0.52, 0.48, 0.62], F 78.92/1.70, P 78.1/ 9.4, R 82.2/11.1 for v43_cv2_k7l1_nonlin_resid',
    'AF1 65.86/2.01 [0.5] 65.11/2.28 [0.48, 0.44, 0.56, 0.52, 0.50], F 78.01/2.40, P 77.0/10.4, R 82.1/12.7 for v43_cv1_k7l1_none___resid',
    'AF1 64.97/1.69 [0.5] 64.87/1.64 [0.46, 0.50, 0.52, 0.52, 0.50], F 78.06/1.62, P 77.6/10.4, R 81.5/12.1 for v43_cv1_k3l2_none___plain',
    'AF1 65.14/2.12 [0.5] 64.81/2.04 [0.52, 0.50, 0.58, 0.56, 0.52], F 78.27/1.95, P 77.5/10.8, R 82.0/11.3 for v43_cv1_k7l1_nonlin_resid',
    'AF1 65.19/1.76 [0.5] 64.55/2.32 [0.42, 0.40, 0.58, 0.44, 0.46], F 78.11/1.73, P 76.8/10.8, R 82.5/11.8 for v43_cv1_k7l1_nonlin_plain',
    'AF1 63.15/4.47 [0.5] 64.35/2.85 [0.20, 0.56, 0.52, 0.48, 0.54], F 78.19/1.65, P 78.4/11.0, R 81.1/12.1 for v43_cv1_k3l2_linear_resid',
    'AF1 64.89/1.67 [0.5] 64.28/1.86 [0.40, 0.46, 0.56, 0.48, 0.54], F 77.51/1.84, P 77.3/11.7, R 81.3/13.0 for v43_cv1_k7l1_linear_resid',
    'AF1 64.69/2.13 [0.5] 64.17/1.72 [0.44, 0.44, 0.56, 0.46, 0.42], F 77.95/2.17, P 76.2/10.2, R 82.8/11.9 for v43_cv1_k3l2_nonlin_resid',
    'AF1 61.79/6.71 [0.5] 63.33/3.88 [0.20, 0.52, 0.50, 0.56, 0.52], F 77.30/2.49, P 79.1/11.2, R 79.1/13.7 for v43_cv1_k3l2_none___resid',
    'AF1 61.94/5.43 [0.5] 63.10/3.65 [0.20, 0.52, 0.50, 0.50, 0.44], F 77.25/2.28, P 77.0/12.4, R 81.4/12.9 for v43_cv1_k7l1_none___plain',
]


def parse_report(report_str):
    original = report_str
    report_str = report_str.split(" ")
    af1_metric = report_str[3].split("/")
    f1_metric = report_str[10].split("/")
    config = report_str[-1].split("_")
    cv_seed = int(config[1][-1])
    model = "_".join(config[2:])
    dispersion_p = float(original.split(",")[6].split("/")[-1])
    dispersion_r = float(original.split(",")[7].split("for")[0].split("/")[-1])
    dispersion = dispersion_p  # np.mean([dispersion_p, dispersion_r])
    output_dict = {
        'af1_mean': float(af1_metric[0]),
        'af1_std':  float(af1_metric[1]),
        'f1_mean': float(f1_metric[0]),
        'dispersion': dispersion,
        'model': model,
        'seed': cv_seed
    }
    return output_dict


def group_by_model(parsed_reports, metric='af1_mean'):
    models = [parsed_report['model'] for parsed_report in parsed_reports]
    models = np.unique(models)
    grouped_reports = {key: [] for key in models}
    for parsed_report in parsed_reports:
        grouped_reports[parsed_report['model']].append(parsed_report[metric])
    return grouped_reports


metric = 'dispersion'
parsed_reports = [parse_report(report) for report in reports]
grouped_reports = group_by_model(parsed_reports, metric=metric)
mean_reports = {model: np.mean(grouped_reports[model]) for model in grouped_reports.keys()}
models = np.array(list(mean_reports.keys()))
mean_af1_list = np.array([mean_reports[model] for model in models])
sorted_locs = np.argsort(-mean_af1_list)
models = models[sorted_locs]
mean_af1_list = mean_af1_list[sorted_locs]
print("Sorted by metric %s" % metric)
for model, af1 in zip(models, mean_af1_list):
    print("%s: %1.4f" % (model, af1))
