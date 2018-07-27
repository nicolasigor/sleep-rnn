from sleep_data_inta import SleepDataINTA
# from sleep_data_mass import SleepDataMASS
# from spindle_detector_fc_v1 import SpindleDetectorFC
from spindle_detector_lstm_v1 import SpindleDetectorLSTM

# Initialize database
dataset = SleepDataINTA()
# dataset = SleepDataMASS()

# Params for detector
params = {
    'sequence': 30,           # Length of CWT sequence in [s]
    'n_chunks': 30,           # Number of chunks for each sequence to perform TBPTT
    'border': 1,              # In [s]. Border to be added to avoid border effects in CWT computation
    'mark_smooth': 1,         # Number of samples to average at the center to get the segment central mark.
    'fs': dataset.get_fs(),   # Sampling frequency of the dataset
    'seq_stride': int(0.05 * dataset.get_fs())  # make prediction every this steps (in samples)
}
# Initialize detector
ss_detector = SpindleDetectorLSTM(params)
# Train detector
max_it = 10
stat_every = 2
save_every = 50000
ss_detector.train(dataset, max_it, stat_every, save_every)
