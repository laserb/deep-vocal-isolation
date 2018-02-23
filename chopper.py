#!/usr/bin/python3
from hashlib import md5
from config import Config


class Chopper(object):
    def __init__(self):
        self.config = Config()
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
