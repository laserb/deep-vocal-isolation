#!/usr/bin/python3
from hashlib import md5
from config import config


class Chopper(object):
    def __init__(self):
        self.config = config
        self.name = self.config.chopname
        self.params = self.config.chopparams

    def get(self):
        function = getattr(self, self.name)
        if self.params:
            params = eval(self.params)

            def chop(matrix):
                return function(matrix, **params)
        else:
            def chop(matrix):
                return function(matrix)
        return chop

    def __hash__(self):
        config = self.name + ":" + self.params
        val = md5(config.encode()).hexdigest()
        return int(val, 16)

    # Slice up matrices into squares
    # so the neural net gets a consistent size for training
    # (doesn't matter for inference)
    def tile(self, matrix, scale):
        slices = []
        for time in range(0, matrix.shape[1] // scale):
            for freq in range(0, matrix.shape[0] // scale):
                s = matrix[freq * scale: (freq + 1) * scale,
                           time * scale: (time + 1) * scale]
                slices.append(s)
        return slices

    def full(self, matrix, scale):
        slices = []
        for time in range(0, matrix.shape[1] // scale):
            s = matrix[1:, time * scale: (time + 1) * scale]
            slices.append(s)
        return slices

    def upper_half(self, matrix, scale):
        slices = []
        half = matrix.shape[0]//2
        for time in range(0, matrix.shape[1] // scale):
            for freq in range(0, half // scale):
                s = matrix[freq * scale: (freq + 1) * scale,
                           time * scale: (time + 1) * scale]
                slices.append(s)
        return slices

    def sliding(self, matrix, scale, step):
        if isinstance(step, int):
            time_step = step
            freq_step = step
        else:
            time_step = step[0]
            freq_step = step[1]
        slices = []
        for time in range(0, (matrix.shape[1] - scale) // time_step):
            for freq in range(0, (matrix.shape[0] - scale) // freq_step):
                s = matrix[freq * freq_step: freq * freq_step + scale,
                           time * time_step: time * time_step + scale]
                slices.append(s)
        return slices

    def infere(self, matrix, scale):
        slices = []
        for time in range(0, matrix.shape[1] // scale + 1):
            s = matrix[0:, time * scale: (time + 1) * scale]
            slices.append(s)
        return slices
