import pyedflib
import numpy as np


def read_eeg(path_edf_file, channel):
    file = pyedflib.EdfReader(path_edf_file)
    signal = file.readSignal(channel)
    fs = file.getSampleFrequency(channel)
    file._close()
    del file
    return signal, fs


def read_marks(path_marks_file, channel):
    marks_file = np.loadtxt(path_marks_file, dtype='i', delimiter=' ')
    marks = marks_file[marks_file[:, 5] == channel][:, [0, 1]]
    return marks


def read_states(path_states_file):
    states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
    # Source format is 1:SQ4  2:SQ3  3:SQ2  4:SQ1  5:REM  6:WA
    # We enforce the fusion of SQ3 and SQ4 in one single stage
    # So now 2:N3  3:N2  4:N1  5:R  6:W
    states[states == 1] = 2
    return states
