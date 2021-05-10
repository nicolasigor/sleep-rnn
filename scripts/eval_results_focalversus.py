import numpy as np


reports = [
    'AF1 62.70/2.52 [0.5] 66.98/1.73 [0.22, 0.24, 0.22, 0.20, 0.26], F 79.89/1.27, P 78.8/ 9.1, R 82.9/ 9.1 for v43_cv2_eps1.0_pos-2',
    'AF1 65.66/1.26 [0.5] 66.90/2.22 [0.36, 0.28, 0.36, 0.38, 0.34], F 79.83/1.48, P 78.7/10.1, R 83.2/ 9.6 for v43_cv2_eps1.0_pos-1',
    'AF1 67.12/3.31 [0.5] 66.70/3.38 [0.50, 0.44, 0.48, 0.50, 0.46], F 79.68/2.89, P 76.4/ 9.4, R 85.0/ 7.7 for v43_cv0_eps1.0_pos+0',
    'AF1 66.19/1.72 [0.5] 66.67/2.01 [0.40, 0.50, 0.40, 0.44, 0.46], F 79.48/1.41, P 78.3/ 9.7, R 82.9/ 9.6 for v43_cv2_eps1.0_pos+0',
    'AF1 67.15/1.42 [0.5] 66.58/1.69 [0.50, 0.48, 0.48, 0.46, 0.48], F 79.70/1.31, P 77.2/ 9.3, R 84.3/ 8.7 for v43_cv2_eps0.2_pos+0',
    'AF1 57.36/3.01 [0.5] 66.57/1.45 [0.16, 0.12, 0.12, 0.16, 0.12], F 79.52/1.48, P 79.4/ 8.6, R 81.6/10.2 for v43_cv2_eps1.0_pos-3',
    'AF1 62.50/1.77 [0.5] 66.32/2.01 [0.42, 0.44, 0.38, 0.40, 0.36], F 79.51/1.57, P 76.2/ 9.7, R 85.3/ 9.1 for v43_cv2_eps0.2_pos-2',
    'AF1 65.03/1.34 [0.5] 66.25/0.58 [0.32, 0.36, 0.38, 0.36, 0.34], F 78.90/1.11, P 77.8/ 9.1, R 82.3/10.8 for v43_cv1_eps1.0_pos-1',
    'AF1 63.18/2.62 [0.5] 66.13/0.91 [0.28, 0.20, 0.26, 0.32, 0.24], F 79.05/0.87, P 77.9/10.3, R 82.8/10.3 for v43_cv1_eps1.0_pos-2',
    'AF1 63.30/4.40 [0.5] 66.11/4.16 [0.40, 0.40, 0.42, 0.42, 0.42], F 79.30/4.21, P 77.8/ 8.8, R 83.1/11.3 for v43_cv0_eps0.2_pos-2',
    'AF1 65.16/3.75 [0.5] 66.05/3.78 [0.46, 0.44, 0.46, 0.44, 0.46], F 79.13/3.41, P 76.8/ 9.8, R 83.9/10.1 for v43_cv0_eps0.2_pos-1',
    'AF1 65.68/4.85 [0.5] 65.93/4.64 [0.46, 0.48, 0.50, 0.50, 0.50], F 79.28/4.09, P 76.7/10.6, R 84.7/10.1 for v43_cv0_eps0.2_pos+0',
    'AF1 56.63/4.37 [0.5] 65.70/4.02 [0.40, 0.32, 0.32, 0.38, 0.30], F 79.12/3.66, P 78.3/ 8.3, R 82.0/11.1 for v43_cv0_eps0.2_pos-3',
    'AF1 65.42/3.89 [0.5] 65.69/4.57 [0.50, 0.34, 0.46, 0.36, 0.42], F 78.65/4.17, P 76.8/ 9.9, R 82.8/ 9.9 for v43_cv0_eps1.0_pos-1',
    'AF1 56.02/2.81 [0.5] 65.69/0.91 [0.36, 0.34, 0.32, 0.32, 0.36], F 79.21/1.13, P 76.0/ 8.3, R 84.7/ 9.9 for v43_cv2_eps0.2_pos-3',
    'AF1 62.47/4.76 [0.5] 65.51/4.11 [0.30, 0.30, 0.20, 0.24, 0.24], F 78.44/3.59, P 77.5/ 9.8, R 81.7/10.5 for v43_cv0_eps1.0_pos-2',
    'AF1 65.56/1.52 [0.5] 65.50/1.53 [0.42, 0.50, 0.50, 0.50, 0.44], F 78.12/1.99, P 77.8/10.4, R 81.6/12.9 for v43_cv1_eps1.0_pos+0',
    'AF1 55.94/6.09 [0.5] 65.49/1.01 [0.34, 0.32, 0.34, 0.38, 0.32], F 78.67/1.30, P 76.9/ 9.9, R 82.9/10.5 for v43_cv1_eps0.2_pos-3',
    'AF1 62.22/2.84 [0.5] 65.40/1.35 [0.38, 0.40, 0.40, 0.40, 0.42], F 78.62/1.67, P 76.2/10.5, R 84.0/10.6 for v43_cv1_eps0.2_pos-2',
    'AF1 56.95/4.59 [0.5] 65.39/4.34 [0.14, 0.14, 0.10, 0.16, 0.16], F 78.82/3.54, P 78.0/10.0, R 82.1/10.3 for v43_cv0_eps1.0_pos-3',
    'AF1 64.89/0.72 [0.5] 65.31/1.38 [0.48, 0.44, 0.44, 0.46, 0.50], F 78.71/1.17, P 75.7/ 9.1, R 84.1/10.0 for v43_cv2_eps0.2_pos-1',
    'AF1 64.77/1.56 [0.5] 65.28/0.99 [0.50, 0.52, 0.50, 0.48, 0.48], F 78.33/1.76, P 77.6/ 9.5, R 81.8/12.1 for v43_cv1_eps0.2_pos+0',
    'AF1 64.04/2.52 [0.5] 64.89/1.61 [0.40, 0.44, 0.48, 0.46, 0.44], F 77.99/2.14, P 76.6/10.2, R 82.5/12.4 for v43_cv1_eps0.2_pos-1',
    'AF1 57.48/5.23 [0.5] 64.75/1.30 [0.12, 0.14, 0.14, 0.10, 0.20], F 77.89/1.21, P 76.5/11.6, R 82.7/12.0 for v43_cv1_eps1.0_pos-3',
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
    precision = float(original.split(",")[6].split("/")[0].split(" ")[-1])
    dispersion_r = float(original.split(",")[7].split("for")[0].split("/")[-1])
    dispersion = np.mean([dispersion_p, dispersion_r])
    output_dict = {
        'af1_mean': float(af1_metric[0]),
        'f1_mean': float(f1_metric[0]),
        'mean_dispersion': dispersion,
        'dispersion_p': dispersion_p,
        'dispersion_r': dispersion_r,
        'precision': precision,
        'model': model,
        'seed': cv_seed
    }
    return output_dict


def group_by_model(parsed_reports, metric):
    models = [parsed_report['model'] for parsed_report in parsed_reports]
    models = np.unique(models)
    grouped_reports = {key: [] for key in models}
    for parsed_report in parsed_reports:
        grouped_reports[parsed_report['model']].append(parsed_report[metric])
    return grouped_reports


metric = 'precision'
larger_is_better = False

parsed_reports = [parse_report(report) for report in reports]
grouped_reports = group_by_model(parsed_reports, metric=metric)
mean_reports = {model: np.mean(grouped_reports[model]) for model in grouped_reports.keys()}
models = np.array(list(mean_reports.keys()))
mean_af1_list = np.array([mean_reports[model] for model in models])
if larger_is_better:
    sorted_locs = np.argsort(-mean_af1_list)
else:
    sorted_locs = np.argsort(mean_af1_list)

sorted_locs = np.argsort(models)

models = models[sorted_locs]
mean_af1_list = mean_af1_list[sorted_locs]
print("Sorted by metric %s" % metric)
for model, af1 in zip(models, mean_af1_list):
    print("%s: %1.2f" % (model, af1))
