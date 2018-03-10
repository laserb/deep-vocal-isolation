#!/usr/bin/python3
from keras import losses
from config import config


class Loss(object):
    def __init__(self):
        self.config = config
        self.loss = self.config.loss

    def get(self):
        return getattr(self, self.loss)

    def mean_squared_error(self, y_true, y_pred):
        return losses.mean_squared_error(y_true, y_pred)

    def mean_absolute_error(self, y_true, y_pred):
        return losses.mean_absolute_error(y_true, y_pred)

    def mean_squared_log_error(self, y_true, y_pred):
        return losses.mean_squared_logarithmic_error(y_true, y_pred)
