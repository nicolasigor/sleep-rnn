"""cwt_ops.py: Module that computes the CWT"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils.constants import CHANNELS_LAST, CHANNELS_FIRST
from utils.constants import ERROR_INVALID


# TODO: change channel ordering for computing the cwt too (not just at the end)
def apply_wavelets(
        inputs,
        wavelets,
        border_crop=0,
        stride=1,
        data_format=CHANNELS_LAST,
        name=None):
    """
    CWT layer implementation in Tensorflow that returns the scalograms tensor.

    Implementation of CWT in Tensorflow, aimed at providing GPU acceleration.
    This layer use computed wavelets. It supports the computation of several
    scalograms. Different scalograms will be stacked along the channel axis.

    Args:
        inputs: (tensor) A batch of 1D tensors of shape [batch_size, time_len].
        wavelets: (list of tuples of arrays) A list of computed wavelet banks.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        data_format: (Optional, {CHANNELS_LAST, CHANNELS_FIRST}, defaults to
            CHANNELS_LAST) Specify the data format of the output data. With the
            default format CHANNELS_LAST, the output has shape
            [batch, signal_size, n_scales, channels]. Alternatively, with the
            format CHANNELS_FIRST, the output has shape
            [batch, channels, signal_size, n_scales].
        name: (Optional, string, defaults to None) A name for the operation.

    Returns:
        Scalogram tensor.
    """
    # Checking
    if data_format not in [CHANNELS_FIRST, CHANNELS_LAST]:
        msg = ERROR_INVALID % (
            [CHANNELS_FIRST, CHANNELS_LAST],
            'data_format', data_format)
        raise ValueError(msg)

    n_scalograms = len(wavelets)

    # Generate the scalograms
    border_crop = int(border_crop/stride)
    start = border_crop
    if border_crop == 0:
        end = None
    else:
        end = -border_crop

    if name is None:
        name = "cwt"
    with tf.variable_scope(name):
        # Reshape input [batch, time_len] -> [batch, 1, time_len, 1]
        inputs_expand = tf.expand_dims(inputs, axis=1)
        inputs_expand = tf.expand_dims(inputs_expand, axis=3)
        scalograms_list = []
        for j in range(n_scalograms):
            with tf.name_scope('%s_%d' % (name, j)):
                bank_real, bank_imag = wavelets[j]  # n_scales filters each
                out_real = tf.nn.conv2d(
                    input=inputs_expand, filter=bank_real,
                    strides=[1, 1, stride, 1], padding="SAME")
                out_imag = tf.nn.conv2d(
                    input=inputs_expand, filter=bank_imag,
                    strides=[1, 1, stride, 1], padding="SAME")
                out_real_crop = out_real[:, :, start:end, :]
                out_imag_crop = out_imag[:, :, start:end, :]
                out_power = tf.sqrt(tf.square(out_real_crop)
                                    + tf.square(out_imag_crop))
                # [batch, 1, time_len, n_scales]->[batch, time_len, n_scales, 1]
                single_scalogram = tf.transpose(out_power, perm=[0, 2, 3, 1])
                scalograms_list.append(single_scalogram)
        # Get all scalograms in shape [batch, time_len, n_scales, n_scalograms]
        scalograms = tf.concat(scalograms_list, -1)
        if data_format == CHANNELS_FIRST:
            # [batch, time_len, n_scales, n_scalograms]
            # -> [batch, time_len, n_scalograms, n_scales]
            scalograms = tf.transpose(scalograms, perm=[0, 3, 1, 2])
    return scalograms