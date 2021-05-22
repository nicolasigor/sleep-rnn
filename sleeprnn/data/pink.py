"""@Author: Nicolas I. Tapia-Rivas"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from sleeprnn.common import constants
from .dataset import Dataset
from .dataset import KEY_EEG, KEY_MARKS
from .dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM
from .utils import apply_lowpass

PATH_PINK_RELATIVE = 'pink'


class Pink(Dataset):
    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        self.channel = 'artificial'
        self.n_signals = 50
        self.n2_id = '2'
        self.unknown_id = '?'
        # Generation parameters
        self.signal_duration = 3 * 3600 + 2 * 20  # 3 hours of useful signal + 1 page at borders
        self.decay_exponent = 1.14  # adjusted in Mass-Train
        self.min_frequency = 1  # [Hz]
        self.max_frequency = 29  # [Hz]
        self.frequency_step = 2  # [Hz]
        self.filter_zero_padding = 10  # [s] removed after generation
        self.signal_standard_deviation = 16.7  # [mu V] adjusted in Mass-Train

        all_ids = np.arange(1, self.n_signals + 1).tolist()
        super(Pink, self).__init__(
            dataset_dir=PATH_PINK_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.PINK_NAME,
            all_ids=all_ids,
            event_name='none',
            hypnogram_sleep_labels=['2'],
            hypnogram_page_duration=[20],
            params=params,
            verbose=verbose
        )
        self.global_std = None
        if verbose:
            print("Global STD", self.global_std)

    def _load_from_source(self):
        n_pages = self.signal_duration // self.page_duration
        data = {}
        start = time.time()
        for i, subject_id in enumerate(self.all_ids):
            print("\nGenerating pink noise ID %s" % subject_id)
            signal = self._generate_pink_noise(subject_id)
            hypnogram = [self.unknown_id] + (n_pages - 2) * [self.n2_id] + [self.unknown_id]
            hypnogram = np.asarray(hypnogram)
            n2_pages = np.where(hypnogram == self.n2_id)[0].astype(np.int16)
            all_pages = np.arange(1, n_pages - 1, dtype=np.int16)
            marks = np.zeros(shape=(0, 2)).astype(np.int32)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Marks SS from E1: %d' % marks.shape[0])
            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                KEY_HYPNOGRAM: hypnogram,
                '%s_1' % KEY_MARKS: marks
            }
            data[subject_id] = ind_dict
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, i + 1, self.n_signals, time.time() - start))
        print('%d records have been read.' % len(data))
        return data

    def _generate_pink_noise(self, seed):
        border_size = int(self.filter_zero_padding * self.fs)
        signal_size = int(self.signal_duration * self.fs)
        extended_size = signal_size + 2 * border_size
        gauss_noise = np.random.RandomState(seed=seed).normal(size=extended_size)
        n_cutoffs = int(np.floor((self.max_frequency - self.min_frequency) / self.frequency_step))
        pink_noise = np.zeros(signal_size)
        for i in range(n_cutoffs + 1):
            cutoff = self.min_frequency + i * self.frequency_step
            narrow_band = apply_lowpass(gauss_noise, self.fs, cutoff)  # (past_f_low, f_low)
            gauss_noise -= narrow_band  # [f_low, fs/2]
            if i > 0:
                narrow_band = narrow_band[border_size:-border_size]
                # Normalize band
                mean = narrow_band.mean()
                std = narrow_band.std()
                narrow_band = (narrow_band - mean) / std
                band_central_frequency = cutoff - self.frequency_step / 2
                narrow_band *= band_central_frequency ** (-self.decay_exponent)
                pink_noise += narrow_band
        mean = pink_noise.mean()
        std = pink_noise.std()
        pink_noise = (pink_noise - mean) / std
        pink_noise *= self.signal_standard_deviation
        pink_noise = pink_noise.astype(np.float32)
        return pink_noise
