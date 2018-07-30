from sleep_data_inta import SleepDataINTA
# from sleep_data_mass import SleepDataMASS
from oldcode.spindle_detector_fc_v1 import SpindleDetectorFC

# Initialize database
dataset = SleepDataINTA()
# dataset = SleepDataMASS()

# Params for detector
params = {
    'context': 0.64,          # Length of CWT in [s]
    'factor_border': 2,       # Signal segment for CWT is (factor_border+1)*context to avoid border effects
    'mark_smooth': 1,         # Number of samples to average at the center to get the segment central mark.
    'fs': dataset.get_fs()    # Sampling frequency of the dataset
}
# Initialize detector
ss_detector = SpindleDetectorFC(params)
# Train detector
max_it = 2
stat_every = 100
save_every = 50000
ss_detector.train(dataset, max_it, stat_every, save_every)
