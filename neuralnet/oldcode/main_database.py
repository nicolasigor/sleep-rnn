# Imports
from __future__ import division
import numpy as np
import pandas as pd
import time

import utils

# Settings

all_names = [
    'ADGU101504',
    'ALUR012904',
    'BECA011405',
    'BRCA062405',
    'BRLO041102',
    'BTOL083105',
    'BTOL090105',
    'CAPO092605',
    'CRCA020205',
    'ESCI031905',
    'TAGO061203']

path_rec = "ssdata_inta/register/"
rec_postamble = ".rec"

path_marks = "ssdata_inta/label/marks/"
marks_preamble = "FixedSS_"
marks_postamble = ".txt"

path_states = "ssdata_inta/label/states/"
states_preamble = "StagesOnly_"
states_postamble = ".txt"

params = {
    'channel': 1,           # Channel to be used
    'dur_epoch': 30,        # Time of window page [s]
    'n2_val': 3,            # N2 state coding value
    'context': 5,           # Context to be added before and after an epoch, in [s]
    'context_add': 1        # Additional context so we can keep the original after FFT, in [s]
    # 'dur_min_ss': 0.3,      # Min SS duration [s]
    # 'dur_max_ss': 3.0       # Max SS duration [s]
}

# Read everything
signal_list = []
marks_list = []
states_list = []
for i in range(len(all_names)):
    # Read EEG Signal
    path_edf_file = path_rec + all_names[i] + rec_postamble
    signal, fs = utils.dataload.read_eeg(path_edf_file, params['channel'])
    signal_list.append(signal)
    # Read Expert marks
    path_marks_file = path_marks + marks_preamble + all_names[i] + marks_postamble
    marks = utils.dataload.read_marks(path_marks_file, params['channel'])
    marks_list.append(marks)
    # Read states
    path_states_file = path_states + states_preamble + all_names[i] + states_postamble
    states = utils.dataload.read_states(path_states_file)
    states_list.append(states)
params['fs'] = fs           # Save sampling frequency [Hz]

print(len(signal_list), ' EEG signals have been read.')
print(len(marks_list), ' sleep spindle marks files have been read.')
print(len(states_list), ' state annotations files have been read.')
print('Sampling Frequency: ', fs, 'Hz')

# Extraction of N2 epochs in data frame of Pandas
n2eeg_df = utils.transform.get_n2_epochs(signal_list, states_list, marks_list, params)
# Clip normalization
n2eeg_df = utils.transform.clip_normalize(n2eeg_df, 99)

start = time.time()

win_size = 0.5 # [s]
step_size = 10 # in samples
min_freq = 0.1 #[hz]
max_freq = 40 #[hz]

win_size = win_size*params['fs']
time_step = 1/params['fs']
win_half = int(np.floor(win_size/2))
n_segment = 2 * win_half + 1
freq = np.fft.fftfreq(n_segment, d=time_step)
freq = freq[0:int(n_segment/2)]
chosen = np.bitwise_and(min_freq <= freq, freq <= max_freq).astype(int)
context_size = params['context_add'] * params['fs']

# First we only split in segments
rows_list = []
size_df = n2eeg_df.shape[0]
for idx in range(size_df):
    if (idx+1) % 50 == 0 and idx != 0:
        print(idx+1, '/', size_df,' -- Time elapsed:', time.time() - start, ' s')
    epoch_signal = n2eeg_df.loc[idx, 'EEG_DATA']
    epoch_marks = n2eeg_df.loc[idx, 'MARKS_DATA']
    n_signal = epoch_signal.size
    for sample in np.arange(context_size, n_signal - context_size, step_size):
        segment = epoch_signal[(sample - win_half):(sample + win_half + 1)]
        # Save it
        dict_tmp = {}
        dict_tmp.update({'ID_REG': n2eeg_df.loc[idx, 'ID_REG']})
        dict_tmp.update({'ID_SEG': n2eeg_df.loc[idx, 'ID_SEG']})
        dict_tmp.update({'ID_EPOCH': n2eeg_df.loc[idx, 'ID_EPOCH']})
        dict_tmp.update({'FFT_DATA': segment})
        dict_tmp.update({'MARK': epoch_marks[sample]})
        rows_list.append(dict_tmp)
print('Splitting Ready, Total Time Elapsed: ', time.time() - start, ' s')
n2fft_df = pd.DataFrame(rows_list)
n2fft_df = n2fft_df[['ID_REG', 'ID_SEG', 'ID_EPOCH', 'FFT_DATA', 'MARK']]
# Now we apply fourier transform


def to_fourier(x):
    fourier = np.abs(np.fft.fft(x * np.hamming(x.size)))
    fourier = fourier[0:int(x.size/2)]
    fourier[1:-2] = 2* fourier[1:-2]
    fourier = fourier[chosen == 1]
    return fourier


n2fft_df.loc[:, 'FFT_DATA'] = n2fft_df.loc[:, 'FFT_DATA'].map(to_fourier)

print('FFT Ready, Total Time Elapsed: ', time.time() - start, ' s')

#Saving
pd.to_pickle(n2fft_df, "pickle_data/n2fft_05_dataframe_full.pkl")  # n2fft_dataframe
print('Saved')
