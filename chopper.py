#!/usr/bin/python3
from hashlib import md5
from config import config
from inspect import signature
import numpy as np
import random


class Chopper(object):
    def __init__(self):
        self.config = config
        self.name = self.config.chopname
        self.params = self.config.chopparams

    def get(self, both=True):
        function = getattr(self, self.name)
        if self.params:
            params = eval(self.params)
        else:
            params = {}
        if both:
            sig = signature(function).parameters
            if 'matrix' in sig:
                def chop_both(mashup, acapella):
                    mashupSlices = function(mashup, **params)
                    acapellaSlices = function(acapella, **params)
                    return mashupSlices, acapellaSlices
            else:
                def chop_both(mashup, acapella):
                    return function(mashup, acapella, **params)
            return chop_both
        else:
            def chop(matrix):
                return function(matrix, **params)
            return chop

    def __hash__(self):
        config = self.name + ":" + self.params
        val = md5(config.encode()).hexdigest()
        return int(val, 16)

    # Slice up matrices into squares
    # so the neural net gets a consistent size for training
    # (doesn't matter for inference)
    def tile(self, matrix, scale, upper=False, **kwargs):
        slices = []
        limit = matrix.shape[0]//2 if upper else matrix.shape[0]

        for time in range(0, matrix.shape[1] // scale):
            for freq in range(0, limit // scale):
                s = matrix[freq * scale: (freq + 1) * scale,
                           time * scale: (time + 1) * scale]
                slices.append(s)
        return slices

    def full(self, matrix, scale, upper=False, **kwargs):
        slices = []
        for time in range(0, matrix.shape[1] // scale):
            if upper:
                s = matrix[0:matrix.shape[0]//2,
                           time * scale: (time + 1) * scale]
            else:
                s = matrix[1:, time * scale: (time + 1) * scale]
            slices.append(s)
        return slices

    def sliding(self, matrix, scale, step, upper=False, **kwargs):
        if isinstance(step, int):
            time_step = step
            freq_step = step
        else:
            time_step = step[0]
            freq_step = step[1]
        slices = []
        limit = matrix.shape[0] // 2 if upper else matrix.shape[0]

        for time in range(0, (matrix.shape[1] - scale) // time_step):
            for freq in range(0, (limit - scale) // freq_step):
                s = matrix[freq * freq_step: freq * freq_step + scale,
                           time * time_step: time * time_step + scale]
                slices.append(s)
        return slices

    def sliding_full(self, matrix, scale, step, upper=False, **kwargs):
        if isinstance(step, int):
            time_step = step
        else:
            time_step = step[0]
        slices = []

        for time in range(0, (matrix.shape[1] - scale) // time_step):
            if upper:
                s = matrix[0:matrix.shape[0] // 2,
                           time * time_step: time * time_step + scale]
            else:
                s = matrix[1:, time * time_step: time * time_step + scale]
            slices.append(s)
        return slices

    def filtered(self, mashup, acapella, scale,
                 upper=False, filter="mean", **kwargs):

        filter_function = getattr(self, "_" + filter)

        mashupSlices = []
        acapellaSlices = []

        limit = acapella.shape[0] // 2 if upper else acapella.shape[0]

        slices = self.tile(acapella, scale, upper)
        mean_deviation = np.sum(slices) / \
            (len(slices) * np.prod(slices[0].shape))

        for time in range(0, acapella.shape[1] // scale):
            for freq in range(0, limit // scale):
                sa = acapella[freq * scale: (freq + 1) * scale,
                              time * scale: (time + 1) * scale]

                sm = mashup[freq * scale: (freq + 1) * scale,
                            time * scale: (time + 1) * scale]

                if mean_deviation < filter_function(sa):
                    acapellaSlices.append(sa)
                    mashupSlices.append(sm)

        return mashupSlices, acapellaSlices

    def filtered_full(self, mashup, acapella, scale,
                      upper=False, filter="mean", **kwargs):

        filter_function = getattr(self, "_" + filter)

        mashupSlices = []
        acapellaSlices = []

        slices = self.tile(acapella, scale, upper)
        mean_deviation = np.sum(slices) / \
            (len(slices) * np.prod(slices[0].shape))

        for time in range(0, acapella.shape[1] // scale):
            if upper:
                sa = acapella[0:acapella.shape[0] // 2,
                              time * scale: (time + 1) * scale]

                sm = mashup[0:mashup.shape[0] // 2,
                            time * scale: (time + 1) * scale]
            else:
                sa = acapella[1:, time * scale: (time + 1) * scale]
                sm = mashup[1:, time * scale: (time + 1) * scale]

            if mean_deviation < filter_function(sa):
                acapellaSlices.append(sa)
                mashupSlices.append(sm)

        return mashupSlices, acapellaSlices

    def random(self, mashup, acapella, scale, iterations,
               upper=False, **kwargs):

        mashupSlices = []
        acapellaSlices = []

        limit = acapella.shape[0] // 2 if upper else acapella.shape[0]

        for i in range(0, iterations):
            random_time = random.randrange(acapella.shape[1] - scale)
            random_freq = random.randrange(limit - scale)

            sa = acapella[random_freq: random_freq + scale,
                          random_time: random_time + scale]

            sm = mashup[random_freq: random_freq + scale,
                        random_time: random_time + scale]

            acapellaSlices.append(sa)
            mashupSlices.append(sm)

        return mashupSlices, acapellaSlices

    def random_full(self, mashup, acapella, scale, slices,
                    upper=False, **kwargs):

        mashupSlices = []
        acapellaSlices = []

        for i in range(0, slices):
            random_time = random.randrange(acapella.shape[1] - scale)

            if upper:
                sa = acapella[0:acapella.shape[0] // 2,
                              random_time: random_time + scale]

                sm = mashup[0:mashup.shape[0] // 2,
                            random_time: random_time + scale]
            else:
                sa = acapella[1:, random_time: random_time + scale]
                sm = mashup[1:, random_time: random_time + scale]

            acapellaSlices.append(sa)
            mashupSlices.append(sm)

        return mashupSlices, acapellaSlices

    def _maximum(self, s):
            return np.max(s)

    def _mean(self, s):
            return np.sum(s) / np.prod(s.shape)

    def infere(self, matrix, scale, **kwargs):
        slices = []
        for time in range(0, matrix.shape[1] // scale + 1):
            s = matrix[0:, time * scale: (time + 1) * scale]
            slices.append(s)
        return slices
