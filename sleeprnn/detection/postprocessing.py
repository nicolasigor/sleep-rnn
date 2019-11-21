from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import find_peaks

from sleeprnn.data.utils import filter_iir_lowpass


def kcomplex_stamp_split(
        signal,
        stamps,
        fs,
        highcut=4,
        left_edge_tol=0.05,
        right_edge_tol=0.2,
        signal_is_filtered=False
):
    left_edge_tol = fs * left_edge_tol
    right_edge_tol = fs * right_edge_tol

    if signal_is_filtered:
        filt_signal = signal
    else:
        filt_signal = filter_iir_lowpass(signal, fs, highcut=highcut)

    new_stamps = []
    for stamp in stamps:
        stamp_size = stamp[1] - stamp[0] + 1
        filt_in_stamp = filt_signal[stamp[0]:stamp[1]]
        negative_peaks, _ = find_peaks(- filt_in_stamp)
        # peaks needs to be negative
        negative_peaks = [
            peak for peak in negative_peaks
            if filt_in_stamp[peak] < 0]

        negative_peaks = [
            peak for peak in negative_peaks
            if left_edge_tol < peak < stamp_size - right_edge_tol]

        n_peaks = len(negative_peaks)
        if n_peaks > 1:
            # Change of sign filtering
            group_peaks = [[negative_peaks[0]]]
            idx_group = 0
            for i in range(1, len(negative_peaks)):
                last_peak = group_peaks[idx_group][-1]
                this_peak = negative_peaks[i]
                signal_between_peaks = filt_in_stamp[last_peak:this_peak]
                min_value = signal_between_peaks.min()
                max_value = signal_between_peaks.max()
                if min_value < 0 < max_value:
                    # there is a change of sign, so it is a new group
                    group_peaks.append([this_peak])
                    idx_group = idx_group + 1
                else:
                    # Now change of sign, same group
                    group_peaks[idx_group].append(this_peak)
            new_peaks = []
            for single_group in group_peaks:
                new_peaks.append(int(np.mean(single_group)))
            negative_peaks = new_peaks

        n_peaks = len(negative_peaks)
        if n_peaks > 1:
            # Split marks
            edges_list = [stamp[0]]
            for i in range(n_peaks-1):
                split_point_rel = (negative_peaks[i] + negative_peaks[i+1]) / 2
                split_point_abs = int(stamp[0] + split_point_rel)
                edges_list.append(split_point_abs)
            edges_list.append(stamp[1])
            for i in range(len(edges_list)-1):
                new_stamps.append([edges_list[i], edges_list[i+1]])
        else:
            new_stamps.append(stamp)
    new_stamps = np.stack(new_stamps, axis=0).astype(np.int32)
    return new_stamps
