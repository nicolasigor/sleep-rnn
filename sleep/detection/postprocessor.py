from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleep.common import checks, constants, pkeys
from sleep.data.stamp_correction import filter_duration_stamps
from sleep.data.stamp_correction import combine_close_stamps
from sleep.data.utils import seq2stamp_with_pages, extract_pages_for_stamps


class PostProcessor(object):

    def __init__(self, event_name, params=None):
        checks.check_valid_value(
            event_name, 'event_name',
            [constants.SPINDLE, constants.KCOMPLEX])

        self.event_name = event_name
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)

    def proba2stamps(
            self,
            pages_sequence,
            pages_indices,
            pages_indices_subset=None,
            thr=0.5):
        """
        If thr is None, pages_sequence is assumed to be already binarized.
        fs_input corresponds to sampling frequency of pages_sequence,
        fs_outputs corresponds to desired sampling frequency.
        """
        # Thresholding
        if thr is None:
            # We assume that sequence is already binary
            pages_sequence_bin = pages_sequence
        else:
            pages_sequence_bin = (pages_sequence >= thr).astype(np.int32)

        # Transformation to stamps
        stamps = seq2stamp_with_pages(
            pages_sequence_bin, pages_indices)

        # Postprocessing
        # Note that when min_separation, min_duration, or max_duration is None,
        # that postprocessing doesn't happen.
        downsampling_factor = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        fs_input = self.params[pkeys.FS] // downsampling_factor
        fs_output = self.params[pkeys.FS]

        if self.event_name == constants.SPINDLE:
            min_separation = self.params[pkeys.SS_MIN_SEPARATION]
            min_duration = self.params[pkeys.SS_MIN_DURATION]
            max_duration = self.params[pkeys.SS_MAX_DURATION]
        else:
            min_separation = self.params[pkeys.KC_MIN_SEPARATION]
            min_duration = self.params[pkeys.KC_MIN_DURATION]
            max_duration = self.params[pkeys.KC_MAX_DURATION]

        stamps = combine_close_stamps(
            stamps, fs_input, min_separation)
        stamps = filter_duration_stamps(
            stamps, fs_input, min_duration, max_duration)

        # Upsampling
        if fs_output > fs_input:
            stamps = self._upsample_stamps(stamps)
        elif fs_output < fs_input:
            raise ValueError('fs_output has to be greater than fs_input')

        if pages_indices_subset is not None:
            page_size = int(self.params[pkeys.PAGE_DURATION] * fs_output)
            stamps = [
                extract_pages_for_stamps(
                    this_stamps, this_pages, page_size)
                for (this_stamps, this_pages)
                in zip(stamps, pages_indices_subset)]

        return stamps

    def proba2stamps_with_list(
            self,
            pages_sequence_list,
            pages_indices_list,
            pages_indices_subset=None,
            thr=0.5):
        stamps_list = [
            self.proba2stamps(
                pages_sequence, pages_indices,
                pages_indices_subset=pages_indices_subset,
                thr=thr)
            for (pages_sequence, pages_indices)
            in zip(pages_sequence_list, pages_indices_list)]
        return stamps_list

    def _upsample_stamps(self, stamps):
        """Upsamples timestamps of stamps to match a greater sampling frequency.
        """
        upsample_factor = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        stamps = stamps * upsample_factor
        stamps[:, 0] = stamps[:, 0] - upsample_factor / 2
        stamps[:, 1] = stamps[:, 1] + upsample_factor / 2
        stamps = stamps.astype(np.int32)
        return stamps
