"""cmorlet.py: Module that implements the CWT using the
complex morlet wavelet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .cwt_ops import apply_wavelets


def compute_cwt(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        flattening=False,
        border_crop=0,
        stride=1,
        trainable=False):
    """Computes the CWT of a batch of signals with the complex Morlet wavelet.
    Please refer to the documentation of compute_wavelets and apply_wavelets to
    see the description of the parameters.
    """
    wavelets, _ = compute_wavelets(
        fb_list=fb_list,
        fs=fs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        n_scales=n_scales,
        flattening=flattening,
        trainable=trainable,
        name='cmorlet')
    cwt = apply_wavelets(
        inputs=inputs,
        wavelets=wavelets,
        border_crop=border_crop,
        stride=stride,
        name='cwt')
    return cwt


def compute_wavelets(
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        flattening=False,
        trainable=False,
        name=None):
    """
    Computes the complex morlet wavelets

    This function computes the complex morlet wavelet defined as:
    PSI(k) = (pi*Fb)^(-0.5) * exp(i*2*pi*Fc*k) * exp(-(k^2)/Fb)
    It supports several values of Fb at once, while Fc is fixed to 1 since we
    can change the frequency of the wavelets by changing the scale. Note that
    greater Fb values will lead to more duration of the wavelet in time,
    leading to better frequency resolution but worse time resolution.
    Scales will be automatically computed from the given frequency range and the
    number of desired scales. The scales  will increase exponentially.

    Args:
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest.
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        flattening: (Optional, boolean, defaults to False) If True, each wavelet
            will be multiplied by its corresponding frequency, to avoid having
            too large coefficients for low frequency ranges, since it is
            common for natural signals to have a spectrum whose power decays
            roughly like 1/f.
        trainable: (Optional, boolean, defaults to False) If True, the fb params
            will be trained with backprop.
        name: (Optional, string, defaults to None) A name for the operation.

    Returns:
        wavelets: (list of tuples of arrays) A list of computed wavelet banks.
        frequencies: (1D array) Array of frequencies for each scale.
    """
    # TODO: Support trainable fb params. Use fb_list as initialization.

    # Checking
    if lower_freq > upper_freq:
        raise ValueError("lower_freq should be lower than upper_freq")
    if lower_freq < 0:
        raise ValueError("Expected positive lower_freq.")

    # Generate initial and last scale
    s_0 = fs / upper_freq
    s_n = fs / lower_freq

    # Generate the array of scales
    base = np.power(s_n / s_0, 1 / (n_scales - 1))
    scales = s_0 * np.power(base, np.arange(n_scales))

    # Generate the frequency range
    frequencies = fs / scales

    # Generate the wavelets
    wavelets = []
    for j, fb in enumerate(fb_list):
        one_side = int(scales[-1] * np.sqrt(5 * fb))
        kernel_size = 2 * one_side + 1
        wavelet_bank_real = np.zeros((1, kernel_size, 1, n_scales))
        wavelet_bank_imag = np.zeros((1, kernel_size, 1, n_scales))
        for i in range(n_scales):
            scale = scales[i]
            k_array = np.arange(kernel_size, dtype=np.float32) - one_side
            norm_constant = np.sqrt(np.pi * fb * scale)
            exp_term = np.exp(-((k_array / scale) ** 2) / fb)
            kernel_base = exp_term / norm_constant
            kernel_real = kernel_base * np.cos(2 * np.pi * k_array / scale)
            kernel_imag = kernel_base * np.sin(2 * np.pi * k_array / scale)
            if flattening:
                kernel_real = kernel_real * frequencies[i]
                kernel_imag = kernel_imag * frequencies[i]
            wavelet_bank_real[0, :, 0, i] = kernel_real
            wavelet_bank_imag[0, :, 0, i] = kernel_imag
        wavelets.append((wavelet_bank_real, wavelet_bank_imag))
    return wavelets, frequencies
