#!/usr/bin/python3
import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from matplotlib.cm import get_cmap
from PIL import Image
import numpy as np
import os

from config import config


class Checkpointer(object):
    def __init__(self, bot):
        self.config = config
        self.checkpoints = self.config.checkpoints
        self.bot = bot

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

    def error_visualization(self):
        return ErrorVisualization(self.bot)


class ErrorVisualization(Callback):
    def __init__(self, bot):
        self.bot = bot

    def on_epoch_end(self, epoch, logs={}):
        xValid = self.bot.xValid
        yValid = self.bot.yValid

        error = np.zeros(yValid[0].shape)

        n = len(xValid) // 100
        for i in range(n):
            y_pred = self.bot.model.predict(xValid[i*100:(i+1)*100],
                                            batch_size=8)
            error += np.sum(np.square(y_pred - yValid[i*100:(i+1)*100]),
                            axis=0)

        error /= (100*n)
        error = error[:, :, 0]
        top_val = np.max(error)
        # scale to range 0, 1
        error /= top_val

        cm_hot = get_cmap('magma')
        im = cm_hot(error)

        # scale to range 0, 255
        im = np.uint8(im * 255)

        im = Image.fromarray(im)
        if not os.path.exists("images"):
            os.mkdir("images")
        im.save("images/error%03d-%f.png" % (epoch, top_val), format='PNG')
