#!/usr/bin/python3
import os
from hashlib import md5


class Config(object):
    def __init__(self):
        self._values = {}
        # Size of FFT windows
        self.fft = self.get_int("FFT", 1536)
        # Path containing training data
        self.data = self.get("DATA", "../bot_data")
        # Proportion of the data to train on
        self.split = self.get_float("SPLIT", 0.9)
        # Number of epochs to train.
        self.epochs = self.get_int("EPOCHS", 10)
        # First epoch number.
        self.start_epoch = self.get_int("START_EPOCH", 0)
        # h5 file to read/write weights to
        self.weights = self.get("WEIGHTS", "weights.h5")
        # Batch size for training
        self.batch = self.get_int("BATCH", 8)
        # directory to store tensorboard log files
        self.logs = self.get("LOGS", "./logs")
        # Phase iterations for reconstruction
        self.phase = self.get_int("PHASE", 10)
        # quit after training for specified epochs
        self.quit = self.get_bool("QUIT", True)
        # Load previous weights file before starting
        self.load = self.get_bool("LOAD", False)
        # size of spectrogram slices to use
        self.slice_size = self.get_int("SLICE_SIZE", 128)
        # train on instrumentals
        self.instrumental = self.get_bool("INSTRUMENTAL", False)

        # Function name for chopping
        self.chopname = self.get("CHOPNAME", "tile")
        # Parameters as a dictionary in string representation.
        # I.e. result of str(mydict).
        self.chopparams = self.get("CHOPPARAMS", "{'scale': 128}")

        # metrics
        self.metrics = self.get("METRICS", "mean_pred,max_pred")

        # Checkpoints run after each epoch
        self.checkpoints = self.get("CHECKPOINTS", "tensorboard,weights")

        # model
        self.model = self.get("MODEL", "acapellabot")
        self.model_params = self.get("MODEL_PARAMS", "")

        # loss
        self.loss = self.get("LOSS", "mean_squared_error")

    def get(self, var, default):
        value = os.environ.get(var, default).strip()
        self._values[var] = value
        return value

    def get_float(self, var, default):
        value = float(os.environ.get(var, default))
        self._values[var] = value
        return value

    def get_int(self, var, default):
        value = int(os.environ.get(var, default))
        self._values[var] = value
        return value

    def get_bool(self, var, default):
        if var in os.environ:
            value = eval(os.environ[var])
        else:
            value = default
        self._values[var] = value
        return value

    def __str__(self):
        result = "# Current configuration\n"
        for name in sorted(self._values):
            result += 'export %s="%s"\n' % (name, self._values[name])
        return result

    def __hash__(self):
        val = md5(str(self).encode()).hexdigest()
        return int(val, 16)


if __name__ == "__main__":
    config = Config()
    print(config)
