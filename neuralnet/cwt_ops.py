from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def complex_morlet_wavelets(
        fc_array,
        fb_array,
        fs,
        lower_freq,
        upper_freq,
        n_scales):
    """
    Computes the complex morlet wavelets

    This function computes the complex morlet wavelet defined as:
    PSI(k) = (pi*Fb)^(-0.5) * exp(i*2*pi*Fc*k) * exp(-(k^2)/Fb)
    It supports several combinations of (Fc, Fb) at once.
    Scales will be automatically computed from the given frequency range and the number of desired scales. The scales
    will increase exponentially.

    Args:
        fc_array: 1D array of values for Fc.
        fb_array: 1D array of values for Fb. Has to be of the same shape as fc_array.
        fs: Sampling frequency of the input.
        lower_freq: Lower frequency to be considered for the scalogram.
        upper_freq: Upper frequency to be considered for the scalogram.
        n_scales: Number of scales to cover the frequency range

    Returns:
        wavelets: A list of computed wavelet banks.
        frequencies: A list of arrays of frequencies for each wavelet bank.
    """
    # Checking
    fc_array = np.array(fc_array)
    fb_array = np.array(fb_array)
    if fc_array.ndim != 1 or fb_array.ndim != 1:
        raise Exception("Expected dimension for fc_array and fb_array is 1")
    elif fc_array.shape != fb_array.shape:
        raise Exception("fc_array and fb_array are expected to have the same shape")
    if lower_freq > upper_freq:
        raise Exception("lower_freq should be lower than upper_freq")
    if lower_freq < 0:
        raise Exception("Expected positive lower_freq.")

    n_scalograms = fc_array[0]

    # Generate initial and last scale
    s_0 = fs / upper_freq
    s_n = fs / lower_freq

    # Generate the array of scales
    base = np.power(s_n / s_0, 1 / (n_scales - 1))
    scales = s_0 * np.power(base, np.arange(n_scales))

    # Generate the wavelets
    wavelets = []
    frequencies = []
    for j in range(n_scalograms):
        fc = fc_array[j]
        fb = fb_array[j]
        wavelet_bank = []
        freq = np.zeros(n_scales)
        for i in range(n_scales):
            scale = scales[i]
            freq[i] = fs / scale
            one_side = int(scale * np.sqrt(5 * fb))
            kernel_size = 2 * one_side + 1
            k_array = np.arange(kernel_size, dtype=np.float32) - one_side
            kernel_base = np.exp(-((k_array / scale) ** 2) / fb) / np.sqrt(np.pi * fb * scale)
            kernel_real = kernel_base * np.cos(2 * np.pi * fc * k_array / scale)
            kernel_imag = kernel_base * np.sin(2 * np.pi * fc * k_array / scale)
            useful_shape = (1, kernel_size, 1, 1)
            kernel_real = np.reshape(kernel_real, useful_shape)
            kernel_imag = np.reshape(kernel_imag, useful_shape)
            kernel_tuple = (kernel_real, kernel_imag)
            wavelet_bank.append(kernel_tuple)
        wavelets.append(wavelet_bank)
        frequencies.append(freq)
    return wavelets, frequencies


def cwt_layer(
        inputs,
        wavelets,
        border_crop=0,
        stride=1,
        frequencies = None,
        flattening=False,
        data_format="channel_last",
        name="cwt"):
    """
    CWT layer implementation in Tensorflow

    Implementation of CWT in Tensorflow, aimed at providing GPU acceleration. This layer use the computed wavelets.
    It supports the computation of several scalograms. Different scalograms will be stacked along the channel dimension.

    Args:
        inputs: 1D tensor of shape [signal_size] or batch of 1D tensors of shape [batch_size, signal_size].
        wavelets: A list of computed wavelet banks.
        border_crop: (Optional) Int>=0 that specifies the number of samples to be removed at each border at the end.
         This parameter allows to input a longer signal than the final desired size to remove border effects of the CWT.
         Default is 0 (no removing at the borders).
        stride: (Optional) Int>0. The stride of the sliding window across the input. Default is 1.
        frequencies: (Optional) A list of arrays of frequencies for each wavelet bank. Defaults to None. If flattening
         is set to True, it is not optional.
        flattening: (Optional) If True, each coefficient will be multiplied by its corresponding frequency, to avoid
         having too large coefficients for low frequency ranges, since it is common for natural signals to have a
         spectrum whose power decays roughly like 1/f. Defaults to True. If set to True, frequencies must be
         provided.
        data_format: (Optional) A string from: "channel_last", "channel_first". Defaults to "channel_last".
         Specify the data format of the output data. With the default format "channel_last", the output has shape
         [batch, n_scales, signal_size, channels]. Alternatively, with the format "channel_first", the output has shape
         [batch, channels, n_scales, signal_size].
        name: (Optional) A name for the operation. Default is "cwt".

    Returns:
        Scalogram tensor.
    """
    # Checking
    if data_format != "channel_first" and data_format != "channel_last":
        raise Exception("Expected 'channel first' or 'channel_last' for data_format")
    if flattening and (frequencies is None):
        raise Exception("If flattening is set to True, frequencies for the wavelets have to be provided.")

    n_scalograms = len(wavelets)
    n_scales = len(wavelets[0])

    # Generate the scalograms
    border_crop = int(border_crop/stride)
    start = border_crop
    if border_crop == 0:
        end = None
    else:
        end = -border_crop
    with tf.variable_scope(name):
        scalograms_list = []
        for j in range(n_scalograms):
            with tf.name_scope(name + "_" + str(j)):
                single_scalogram_list = []
                for i in range(n_scales):
                    kernel_tuple = wavelets[j][i]
                    kernel_real = tf.constant(kernel_tuple[0], dtype=tf.float32)
                    kernel_imag = tf.constant(kernel_tuple[1], dtype=tf.float32)
                    out_real = tf.nn.conv2d(input=inputs, filter=kernel_real,
                                            strides=[1, 1, stride, 1], padding="SAME")
                    out_imag = tf.nn.conv2d(input=inputs, filter=kernel_imag,
                                            strides=[1, 1, stride, 1], padding="SAME")
                    out_real_crop = out_real[:, :, start:end, :]
                    out_imag_crop = out_imag[:, :, start:end, :]
                    out_power = tf.sqrt(tf.square(out_real_crop) + tf.square(out_imag_crop))
                    if flattening:
                        this_freq = frequencies[j][i]
                        out_power = out_power * this_freq
                    single_scalogram_list.append(out_power)
                single_scalogram = tf.concat(single_scalogram_list, 1)
                scalograms_list.append(single_scalogram)
        scalograms = tf.concat(scalograms_list, -1)

        if data_format == "channel_first":
            scalograms = tf.transpose(scalograms, perm=[0, 3, 1, 2])
    return scalograms
