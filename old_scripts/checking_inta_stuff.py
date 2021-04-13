import numpy as np
import pyedflib

import os

inta_path = os.path.join('..', 'resources/datasets/inta')
rec_folder = os.path.join(inta_path, 'register')
channel = 0
registers = os.listdir(rec_folder)
registers.sort()
for reg in registers:
    name = reg.split(".")[0]
    reg_path = os.path.join(rec_folder, reg)
    print("\n", reg_path)

    with pyedflib.EdfReader(reg_path) as file:
        signal = file.readSignal(channel)
        fs_old = file.samplefrequency(channel)
        fs_old = int(fs_old)
        # Check
        print('Channel extracted: %s at fs %s Hz' % (
            file.getLabel(channel), fs_old))
    print("Signal duration %1.3f h" % (signal.size / (fs_old * 3600)))
    print("Signal duration divided by 30s: %s" % (signal.size / (fs_old * 30)))
    states = np.loadtxt(os.path.join(inta_path, 'label/state', 'StagesOnly_%s.txt' % name))
    n_pages = states.size
    print("There are %d pages of 30s" % n_pages)
    print("Last page label:", states[-1])
