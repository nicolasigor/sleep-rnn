import numpy as np


reports = [
    'AF1 66.70/2.36 [0.5] 66.63/2.43 [0.50, 0.44, 0.38, 0.38, 0.48], F 79.70/1.94, P 76.8/ 9.2, R 84.7/ 9.3 for v43_cv2_eps0.5_pos-1',
    'AF1 66.51/1.75 [0.5] 66.38/1.89 [0.48, 0.50, 0.48, 0.46, 0.48], F 79.32/1.58, P 77.5/ 9.8, R 83.5/10.2 for v43_cv2_eps0.5_pos+0',
    'AF1 66.07/3.37 [0.5] 66.29/3.53 [0.46, 0.48, 0.46, 0.46, 0.40], F 79.36/3.22, P 76.5/ 9.3, R 84.4/ 8.6 for v43_cv0_eps0.5_pos-1',
    'AF1 62.63/4.30 [0.5] 66.15/3.98 [0.36, 0.36, 0.32, 0.34, 0.36], F 79.23/3.61, P 76.9/ 9.1, R 83.6/ 8.9 for v43_cv0_eps0.5_pos-2',
    'AF1 65.96/4.14 [0.5] 66.07/4.19 [0.48, 0.46, 0.48, 0.50, 0.52], F 79.32/3.36, P 77.7/10.0, R 83.2/ 9.2 for v43_cv0_eps0.5_pos+0',
    'AF1 62.53/2.23 [0.5] 65.84/1.88 [0.38, 0.38, 0.34, 0.34, 0.36], F 78.87/1.12, P 77.9/10.3, R 82.4/10.9 for v43_cv2_eps0.5_pos-2',
    'AF1 65.04/1.66 [0.5] 65.42/1.28 [0.48, 0.50, 0.50, 0.46, 0.44], F 78.62/1.77, P 77.9/10.5, R 82.2/11.5 for v43_cv1_eps0.5_pos+0',
    'AF1 62.02/3.93 [0.5] 65.36/1.81 [0.32, 0.40, 0.42, 0.32, 0.34], F 78.81/1.70, P 77.2/10.3, R 83.2/11.0 for v43_cv1_eps0.5_pos-2',
    'AF1 64.70/2.16 [0.5] 65.29/1.50 [0.38, 0.42, 0.46, 0.46, 0.42], F 78.47/2.55, P 77.1/ 9.4, R 82.6/12.1 for v43_cv1_eps0.5_pos-1',
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
