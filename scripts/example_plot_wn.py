import os
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.data.loader import load_dataset
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset

SEED_LIST = [123, 234, 345, 456]


dataset_name = 'mass_ss'
task_mode = 'wn'
which_expert = 1
seed = 0

dataset = load_dataset(dataset_name)
all_train_ids = dataset.train_ids
_, val_ids = utils.split_ids_list(all_train_ids, seed=SEED_LIST[seed])
data_val = FeederDataset(dataset, val_ids, task_mode, which_expert=which_expert)
available_ids = data_val.get_ids()

print('Available IDs', available_ids)

for sub_id in available_ids:
    print('Chosen ID %d' % sub_id)

    # Load sample prediction
    filename = os.path.join(
        project_root,
        'results',
        'predictions_%s' % dataset_name,
        '20190504_bsf_%s_train_%s' % (task_mode, dataset_name),
        'bsf',
        'seed%d' % seed,
        'prediction_%s_val.pkl' % task_mode
    )
    with open(filename, 'rb') as handle:
        my_data = pickle.load(handle)

    # Probability
    my_proba = my_data.probabilities_dict[sub_id]
    my_proba_time = np.arange(my_proba.shape[0]) / 25

    # N2
    n2_pages = my_data.get_subject_pages(sub_id, 'n2', False)
    n2_pages_seq = np.zeros(n2_pages[-1]+1)
    n2_pages_seq[n2_pages] = 1
    n2_pages_time = np.arange(n2_pages_seq.shape[0]) * 20

    # Expert stamps
    my_expert = data_val.get_subject_stamps(sub_id, which_expert)
    my_expert = utils.stamp2seq(my_expert, 0, my_proba.shape[0] * 8 + 1)
    my_expert_time = np.arange(my_expert.shape[0]) / 200

    # Plot
    my_color = '#5d6d7e'
    fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True, dpi=200)
    ax[0].fill_between(
        n2_pages_time,
        n2_pages_seq, 0,
        label='N2 state S%02d' % sub_id,
        facecolor=my_color
    )
    ax[0].set_yticks([0, 1])
    ax[0].legend(loc='upper left', fontsize=8)
    ax[0].set_title('MASS-SS-WN-Validation (Seed 123)')

    ax[1].plot(
        my_proba_time,
        my_proba,
        label='Proba S%02d' % sub_id,
        color=my_color
    )
    ax[1].set_yticks([0, 1])
    ax[1].legend(loc='upper left', fontsize=8)

    ax[2].plot(
        my_expert_time,
        my_expert,
        label='Expert S%02d' % sub_id,
        color=my_color
    )
    ax[2].set_yticks([0, 1])
    ax[2].set_xlabel('Time [s]')
    ax[2].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()
