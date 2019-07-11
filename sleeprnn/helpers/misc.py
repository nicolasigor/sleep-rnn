def get_inta_eeg_names(signal_names):
    result = []
    for single_name in signal_names:
        if len(single_name) == 5 and single_name[2] == '-':
            result.append(single_name)
    return result


def get_inta_eog_emg_names(signal_names):
    result = []
    # Look for EOG
    possible_names = [
        'MOR',
        'ojo izquierdo'
    ]
    for single_name in signal_names:
        if single_name in possible_names:
            result.append(single_name)
            break

    # Look for EMG
    possible_names = [
        'EMG',
        'EMG menton'
    ]
    for single_name in signal_names:
        if single_name in possible_names:
            result.append(single_name)
            break
    return result
