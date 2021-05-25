from pprint import pprint
import re

import numpy as np


def get_reports():
    fname = "reports_cvseeds.txt"
    with open(fname, 'r') as file:
        reports = file.read()
    reports = reports.split("\n")
    header = reports[0]
    reports = reports[1:]
    reports = [r for r in reports if r]
    return header, reports


def parse_report(report_str, model_loc, cv_seed_loc):
    report_str = re.sub("\[(.+?)\]", "", report_str)
    report_str = re.sub(",", "", report_str)
    report_str = re.sub("/ ", "/", report_str)
    report_str = re.sub('\s+', ' ', report_str)
    report_list = report_str.split(" ")
    output_dict = {
        'model': report_list[-1].split("_")[model_loc],
        'seed': int(report_list[-1].split("_")[cv_seed_loc][-1]),
        'af1_mean': report_list[2].split("/")[0],
        'af1_std': report_list[2].split("/")[1],
        'f1_mean': report_list[4].split("/")[0],
        'f1_std': report_list[4].split("/")[1],
        'p_mean': report_list[6].split("/")[0],
        'p_std': report_list[6].split("/")[1],
        'r_mean': report_list[8].split("/")[0],
        'r_std': report_list[8].split("/")[1],
    }
    for key in output_dict.keys():
        if key not in ['model', 'seed']:
            output_dict[key] = float(output_dict[key])
    output_dict['dispersion'] = (output_dict['p_std'] + output_dict['r_std']) / 2
    return output_dict


def group_by_model(parsed_reports, metric):
    models = [parsed_report['model'] for parsed_report in parsed_reports]
    models = np.unique(models)
    grouped_reports = {key: [] for key in models}
    for parsed_report in parsed_reports:
        grouped_reports[parsed_report['model']].append(parsed_report[metric])
    return grouped_reports


if __name__ == "__main__":
    model_loc = 2
    cv_seed_loc = 1
    metric = 'p_std'
    larger_is_better = True
    sort_by_value = False

    header, reports = get_reports()
    parsed_reports = [parse_report(report, model_loc, cv_seed_loc) for report in reports]
    grouped_reports = group_by_model(parsed_reports, metric=metric)
    mean_reports = {model: np.mean(grouped_reports[model]) for model in grouped_reports.keys()}
    models = np.array(list(mean_reports.keys()))
    mean_metric = np.array([mean_reports[model] for model in models])
    print(header)
    if sort_by_value:
        if larger_is_better:
            sorted_locs = np.argsort(-mean_metric)
        else:
            sorted_locs = np.argsort(mean_metric)
    else:
        sorted_locs = np.argsort(models)
    models = models[sorted_locs]
    mean_metric = mean_metric[sorted_locs]
    print("Grouped result for metric %s" % metric)
    for model, metric_value in zip(models, mean_metric):
        print("%20s: %1.2f" % (model, metric_value))
