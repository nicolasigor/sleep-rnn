from sleep_data_inta import SleepDataINTA
from detector_fc_v1 import DetectorFC
# from detector_fc_pywavelet import DetectorFCPy

# Initialize database
dataset = SleepDataINTA()
# Params for detector
params = {
    'context': 1.28,          # Length of CWT in [s]
    'factor_border': 2,       # Signal segment for CWT is (factor_border+1)*context to avoid border effects
    'mark_smooth': 1,         # Number of samples to average at the center to get the segment central mark.
    'fs': dataset.fs          # Sampling frequency of the dataset
}
# Initialize detector
ss_detector = DetectorFC(params)
# ss_detector = DetectorFCPy(params)
# Train detector
max_it = 250000
stat_every = 100
save_every = 50000
ss_detector.train(dataset, max_it, stat_every, save_every)
