from oldcode.ss_detector_fc_2 import Detector
import time

# Settings

all_names = [
    'ADGU101504',
    'ALUR012904',
    # 'BECA011405',  we will skip this one for now
    'BRCA062405',
    'BRLO041102',
    'BTOL083105',
    'BTOL090105',
    'CAPO092605',
    'CRCA020205',
    'ESCI031905',
    'TAGO061203']

path_rec = "ssdata/register/"
rec_postamble = ".rec"

path_marks = "ssdata/label/marks/"
marks_preamble = "FixedSS_"
marks_postamble = ".txt"

path_states = "ssdata/label/states/"
states_preamble = "StagesOnly_"
states_postamble = ".txt"

params = {
    'channel': 1,             # Channel to be used
    'dur_epoch': 30,          # Time of window page [s]
    'n2_val': 3,              # N2 state coding value
    'context': 1.28,          # Length of context for timestep, in [s]
    'factor_border': 2,
    # spectrogram will be computed in a segment of length (factor_border+1)*context to avoid border effects
    'mark_smooth': 1,          # Number of samples to average at the center to get the segment central mark.
    'percentile': 99,   # percentil for clipping
    'fs': 200  # Sampling frequency of the dataset
}

# Build list of paths
start = time.time()
data_path_list = []
for i in range(len(all_names)):
    path_edf_file = path_rec + all_names[i] + rec_postamble
    path_marks_file = path_marks + marks_preamble + all_names[i] + marks_postamble
    path_states_file = path_states + states_preamble + all_names[i] + states_postamble
    # Save data
    ind_dict = {'file_edf': path_edf_file,
                'file_marks': path_marks_file,
                'file_states': path_states_file}
    data_path_list.append(ind_dict)
print(len(data_path_list), ' records in dataset.')
print('Total Time: ' + str(time.time() - start) + ' [s]')

# Split in train, val and test
random_perm = [4, 7, 3, 8, 5, 6, 2, 0, 1, 9]
test_idx = random_perm[0:2]
val_idx = random_perm[2:4]
train_idx = random_perm[4:]

train_path_list = [data_path_list[i] for i in train_idx]
val_path_list = [data_path_list[i] for i in val_idx]
test_path_list = [data_path_list[i] for i in test_idx]

print('Training set size:', len(train_path_list), '-- Records:', train_idx)
print('Validation set size:', len(val_path_list), '-- Records:', val_idx)
print('Test set size:', len(test_path_list), '-- Records:', test_idx)

# Train detector
ss_detector = Detector(params, train_path_list, val_path_list)
max_it = 100
stat_every = 20
save_every = 200
ss_detector.train(max_it, stat_every, save_every)
