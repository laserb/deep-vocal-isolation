#!/usr/bin/python3
"""
Normalizer class

Providides normalization functionalities to be used for data preparation

"""

import numpy as np
import copy
from hashlib import md5

from config import config


class Normalizer(object):
    def __init__(self):
        self.config = config
        self.normalizer = self.config.normalizer
        self.params = self.config.normalizer_params

    def get(self, both=True):
        function = getattr(self, self.normalizer)
        if self.params:
            params = eval(self.params)
        else:
            params = {}
        if both:
            def normalize_all(in_mashup, in_vocal):
                mashup = copy.deepcopy(in_mashup)
                vocal = copy.deepcopy(in_vocal)
                for i in range(len(mashup)):
                    mashup[i], norm = function(mashup[i], **params)
                    vocal[i], _ = function(vocal[i], norm=norm, **params)
                return mashup, vocal
            return normalize_all
        else:
            def normalize(matrix, norm=None):
                return function(copy.deepcopy(matrix), norm=norm, **params)
            return normalize

    def get_reverse(self):
        function = getattr(self, "reverse_%s" % self.normalizer)

        def denormalize(matrix, norm):
            return function(matrix.copy(), norm)
        return denormalize

    def __hash__(self):
        config = self.normalizer + ":" + self.params
        val = md5(config.encode()).hexdigest()
        return int(val, 16)

    def dummy(self, matrix, norm=None):
        return matrix, 1

    def reverse_dummy(self, matrix, norm):
        return matrix

    def percentile(self, matrix, percentile, median=True, norm=None):
        if norm is not None:
            if self.config.learn_phase:
                ((shift_real, scale_real),
                 (shift_imag, scale_imag)) = norm
                matrix[:, :, 0] -= shift_real
                matrix[:, :, 0] /= scale_real

                matrix[:, :, 1] -= shift_imag
                matrix[:, :, 1] /= scale_imag

                matrix = np.clip(matrix, -1, 1)
            else:
                matrix /= norm
            return matrix, norm
        else:
            if self.config.learn_phase:
                if median:
                    # shift median to zero
                    shift_real = np.median(matrix[:, :, 0])
                    shift_imag = np.median(matrix[:, :, 1])
                else:
                    # shift mean to zero
                    shift_real = np.mean(matrix[:, :, 0])
                    shift_imag = np.mean(matrix[:, :, 1])
                matrix[:, :, 0] -= shift_real
                matrix[:, :, 1] -= shift_imag

                scale_real = np.percentile(np.abs(matrix[:, :, 0]), percentile)
                scale_imag = np.percentile(np.abs(matrix[:, :, 1]), percentile)

                # do not scale to range 0 - 1,
                # if most of the data is close to 0
                if scale_real < 10e-5:
                    scale_real = 1
                if scale_imag < 10e-5:
                    scale_imag = 1

                norm = ((shift_real, scale_real),
                        (shift_imag, scale_imag))

                matrix[:, :, 0] /= scale_real
                matrix[:, :, 1] /= scale_imag

                matrix = np.clip(matrix, -1, 1)

                return matrix, norm
            else:
                norm = np.percentile(matrix, percentile)
                # do not scale to range 0 - 1,
                # if most of the data is close to 0
                if norm < 10e-5:
                    norm = 1
                matrix /= norm

        return matrix, norm

    def reverse_percentile(self, matrix, norm):
        if self.config.learn_phase:
            ((shift_real, scale_real),
             (shift_imag, scale_imag)) = norm
            matrix[:, :, 0] *= scale_real
            matrix[:, :, 0] += shift_real
            matrix[:, :, 1] *= scale_imag
            matrix[:, :, 1] += shift_imag
        else:
            matrix *= norm
        return matrix
