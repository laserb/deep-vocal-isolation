#!/usr/bin/python3
import keras.backend as K
from config import config


class Metrics(object):
    def __init__(self):
        self.config = config
        self.metrics = self.config.metrics

    def get(self):
        if not self.metrics:
            return []
        names = self.metrics.split(",")
        metrics = []
        for name in names:
            metrics.append(getattr(self, name.strip()))
        return metrics

    def mean_pred(self, y_true, y_pred):
        return K.mean(y_pred)

    def max_pred(self, y_true, y_pred):
        return K.max(y_pred)
