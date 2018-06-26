import numpy as np
import pandas as pd


def seq2inter(sequence):
    if not np.array_equal(sequence, sequence.astype(bool)):
        raise Exception('Sequence must have binary values only')
    interval = []
    n = len(sequence)
    prev_val = 0
    for i in range(n):
        if sequence[i] > prev_val:      # We just turned on
            interval.append([i, i])
        elif sequence[i] < prev_val:    # We just turned off
            interval[-1][1] = i-1
        prev_val = sequence[i]
    if sequence[-1] == 1:
        interval[-1][1] = n-1
    interval = np.stack(interval)
    return interval


def inter2seq(inter, start, end):
    if (inter < start).sum() > 0 or (inter > end).sum() > 0:
        raise Exception('Values in inter matrix should be within start and end bounds')
    sequence = np.zeros(end - start + 1).astype(int)
    for i in range(len(inter)):
        start_sample = inter[i, 0] - start - 1
        end_sample = inter[i, 1] - start
        sequence[start_sample:end_sample] = 1
    return sequence


def get_n2_epochs(signal_list, states_list, marks_list, params):
    # Extraction Step
    # For now, drop epoch if it is at the very beginning or at the very end of a register, because we can't have context
    # And if it is at the end, we can have a different length, which is beyond the scope for now
    rows_list = []
    for ind in range(len(signal_list)):
        signal = signal_list[ind]
        states = states_list[ind]
        marks = marks_list[ind]
        marks = inter2seq(marks, 0, len(signal) - 1)
        n_epoch = len(states)
        # Find n2 epochs segments
        n2_epochs = (states == params['n2_val']).astype(int)
        n2_epochs = seq2inter(n2_epochs)
        for seg in range(len(n2_epochs)):
            for epoch in range(n2_epochs[seg, 0], n2_epochs[seg, 1] + 1):
                if epoch == 0 or epoch == n_epoch - 1:
                    # If this is the first or the last epoch, drop it
                    continue
                dict_tmp = {}
                dict_tmp.update({'ID_REG': ind})
                dict_tmp.update({'ID_SEG': seg})
                dict_tmp.update({'ID_EPOCH': epoch})
                # Now get EEG data
                sample_start = (epoch * params['dur_epoch'] - params['context']) * params['fs']
                sample_end = ((epoch + 1) * params['dur_epoch'] + params['context']) * params['fs']
                dict_tmp.update({'EEG_DATA': signal[sample_start:sample_end]})
                # Now get marks
                dict_tmp.update({'MARKS_DATA': marks[sample_start:sample_end]})
                rows_list.append(dict_tmp)
    df = pd.DataFrame(rows_list, columns=rows_list[0].keys())
    return df


def clip_normalize(df, percentile):
    # Normalization step
    for ind in range(len(df.ID_REG.unique())):
        # Find data from this register
        data = df.loc[df['ID_REG'] == ind, 'EEG_DATA']
        data = np.concatenate(data.values)
        # Found 99th percentile so we can clip, and compute Z-core only with data below that thr
        clipnorm = get_clipnorm(data, percentile)
        # Now we clip and normalize
        df.loc[df['ID_REG'] == ind, 'EEG_DATA'] = df.loc[df['ID_REG'] == ind, 'EEG_DATA'].map(clipnorm)
    return df


def get_clipnorm(data, percentil_value):
    thr = np.percentile(np.abs(data), percentil_value)
    data[np.abs(data) > thr] = float('nan')
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)

    def clipnorm(x):
        return (np.clip(x, -thr, thr) - data_mean) / data_std

    return clipnorm
