import numpy as np


reports = [
    'AF1 66.29/3.81 [0.5] 66.79/3.44 [0.44, 0.46, 0.50, 0.46, 0.64], F 79.50/2.93, P 77.8/ 9.6, R 83.3/ 9.1 for v43_cv0_lstm',
    'AF1 66.74/1.79 [0.5] 66.64/1.89 [0.52, 0.46, 0.44, 0.46, 0.48], F 79.66/1.55, P 77.7/ 9.0, R 83.6/ 9.3 for v43_cv2_lstm_skip_early',
    'AF1 66.73/1.40 [0.5] 66.64/1.62 [0.50, 0.48, 0.42, 0.40, 0.54], F 79.42/1.71, P 79.7/ 8.8, R 81.4/11.4 for v43_cv2_lstm',
    'AF1 66.08/1.79 [0.5] 66.35/1.56 [0.52, 0.48, 0.36, 0.52, 0.52], F 79.62/1.20, P 78.6/ 8.9, R 82.5/ 9.4 for v43_cv2_lstm_skip_late',
    'AF1 65.25/4.60 [0.5] 66.01/3.91 [0.38, 0.46, 0.54, 0.50, 0.60], F 78.93/3.35, P 76.4/10.6, R 84.1/ 9.1 for v43_cv0_lstm_skip_late',
    'AF1 65.53/3.30 [0.5] 65.69/2.46 [0.52, 0.50, 0.54, 0.42, 0.46], F 78.26/3.27, P 78.8/ 9.5, R 80.7/13.2 for v43_cv1_lstm',
    'AF1 65.90/2.52 [0.5] 65.65/1.47 [0.48, 0.46, 0.60, 0.40, 0.50], F 78.70/1.88, P 77.3/10.1, R 83.0/11.4 for v43_cv1_lstm_skip_early',
    'AF1 65.45/4.43 [0.5] 65.61/4.36 [0.50, 0.44, 0.50, 0.54, 0.60], F 78.49/3.88, P 77.9/10.5, R 81.8/11.3 for v43_cv0_lstm_skip_early',
    'AF1 65.54/1.59 [0.5] 65.00/1.80 [0.48, 0.48, 0.56, 0.56, 0.48], F 77.84/2.44, P 76.8/10.0, R 82.1/13.4 for v43_cv1_lstm_skip_late'
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
    dispersion = np.mean([dispersion_p, dispersion_r])
    output_dict = {
        'af1_mean': float(af1_metric[0]),
        'f1_mean': float(f1_metric[0]),
        'mean_dispersion': dispersion,
        'dispersion_p': dispersion_p,
        'dispersion_r': dispersion_r,
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


metric = 'dispersion_r'
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
models = models[sorted_locs]
mean_af1_list = mean_af1_list[sorted_locs]
print("Sorted by metric %s" % metric)
for model, af1 in zip(models, mean_af1_list):
    print("%s: %1.4f" % (model, af1))
