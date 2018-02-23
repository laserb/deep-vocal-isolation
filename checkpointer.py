#!/usr/bin/python3
import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard
from config import Config


class Checkpointer(object):
    def __init__(self):
        self.config = Config()
        self.checkpoints = self.config.checkpoints

    def get(self):
        if not self.checkpoints:
            return []
        names = self.checkpoints.split(",")
        checkpoints = []
        for name in names:
            checkpoints.append(getattr(self, name.strip())())
        return checkpoints

    def tensorboard(self):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_dir = self.config.logs + "/{}".format(date)
        return TensorBoard(log_dir=log_dir)

    def weights(self):
        return ModelCheckpoint(filepath='./weights.hdf5',
                               verbose=1,
                               save_best_only=True)
