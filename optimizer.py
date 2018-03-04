#!/usr/bin/python3
from config import config
from keras.optimizers import Adam, RMSprop


class Optimizer(object):
    def __init__(self):
        self.config = config
        self.optimizer = self.config.optimizer
        self.params = self.config.optimizer_params

    def get(self):
        if self.params:
            params = eval(self.params)
        else:
            params = {}
        return getattr(self, self.optimizer)(**params)

    def adam(self, **kwargs):
        return Adam(**kwargs)

    def rmsprop(self, **kwargs):
        return RMSprop(**kwargs)
