from . import misc
from sleeprnn.common import constants


def print_available_ckpt(optimal_thr_for_ckpt_dict, filter_dates):
    if filter_dates[0] is None:
        filter_dates[0] = -1
    if filter_dates[1] is None:
        filter_dates[1] = 1e12
    print('Available ckpt:')
    for key in optimal_thr_for_ckpt_dict.keys():
        key_date = int(key.split("_")[0])
        if filter_dates[0] <= key_date <= filter_dates[1]:
            print('    %s' % key)


def print_performance_at_iou(performance_data_dict, iou_thr, label):
    iou_curve_axis = performance_data_dict[constants.IOU_CURVE_AXIS]
    idx_to_show = misc.closest_index(iou_thr, iou_curve_axis)

    mean_f1_vs_iou = performance_data_dict[constants.F1_VS_IOU].mean(axis=0)
    std_f1_vs_iou = performance_data_dict[constants.F1_VS_IOU].std(axis=0)
    msg = 'F1 %1.2f/%1.2f' % (
        100*mean_f1_vs_iou[idx_to_show], 100*std_f1_vs_iou[idx_to_show])

    mean_rec_vs_iou = performance_data_dict[constants.RECALL_VS_IOU].mean(axis=0)
    std_rec_vs_iou = performance_data_dict[constants.RECALL_VS_IOU].std(axis=0)
    msg = msg + ', Recall %1.2f/%1.2f' % (
        100*mean_rec_vs_iou[idx_to_show], 100*std_rec_vs_iou[idx_to_show])

    mean_pre_vs_iou = performance_data_dict[constants.PRECISION_VS_IOU].mean(
        axis=0)
    std_pre_vs_iou = performance_data_dict[constants.PRECISION_VS_IOU].std(axis=0)
    msg = msg + ', Precision %1.2f/%1.2f' % (
        100*mean_pre_vs_iou[idx_to_show], 100*std_pre_vs_iou[idx_to_show])

    msg = msg + ', AF1 %1.2f/%1.2f' % (
        100*performance_data_dict[constants.MEAN_AF1].mean(),
        100*performance_data_dict[constants.MEAN_AF1].std())
    msg = msg + ', IoU %1.2f/%1.2f' % (
        100*performance_data_dict[constants.MEAN_IOU].mean(),
        100*performance_data_dict[constants.MEAN_IOU].std()
    )

    msg = msg + ' for %s' % label
    print(msg)
