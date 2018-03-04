#!/usr/bin/python3
import os
from hashlib import md5
import datetime


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
        self.tensorboard = self.get("TENSORBOARD", "./tensorboard")
        # verbosity level of the tensorboard checkpoint
        self.tensorboard_info = self.get("TENSORBOARD_INFO", "default")
        # early stopping checkpoint parameters
        self.early_stopping = self.get("EARLY_STOPPING",
                                       "{'min_delta': 0.001, 'patience': 3}")
        # Phase iterations for reconstruction
        self.phase = self.get_int("PHASE", 10)
        # quit after training for specified epochs
        self.quit = self.get_bool("QUIT", True)
        # Load previous weights file before starting
        self.load = self.get_bool("LOAD", False)

        # The size of the slices for the inference
        self.inference_slice = self.get_int("INFERENCE_SLICE", 3500)

        # train on instrumentals
        self.instrumental = self.get_bool("INSTRUMENTAL", False)

        # Function name for chopping
        self.chopname = self.get("CHOPNAME", "tile")
        # Parameters as a dictionary in string representation.
        # I.e. result of str(mydict).
        self.chopparams = self.get("CHOPPARAMS", "{'scale': 128, "
                                                 "'step': 64, 'slices':256, "
                                                 "'upper':False, "
                                                 "'filter':'maximum'}")

        # metrics
        self.metrics = self.get("METRICS", "mean_pred,max_pred")

        # Checkpoints run after each epoch
        self.checkpoints = self.get("CHECKPOINTS", "tensorboard,weights")

        # model
        self.model = self.get("MODEL", "leaky_dropout")
        self.model_params = self.get("MODEL_PARAMS", "{"
                                     "'alpha1': 0.1,"
                                     "'alpha2': 0.01,"
                                     "'rate': 0.1"
                                     "}")

        # batch generator
        self.batch_generator = self.get("BATCH_GENERATOR", "random")
        self.epoch_steps = self.get_int("EPOCH_STEPS", 50000)

        # loss
        self.loss = self.get("LOSS", "mean_squared_error")

        # optimizer
        self.optimizer = self.get("OPTIMIZER", "adam")
        self.optimizer_params = self.get("OPTIMIZER_PARAMS", "")

        # directory for analysis files
        self.analysis_path = self.get("ANALYSIS_PATH", "./analysis")

        # percentile normalization
        self.normalizer = self.get("NORMALIZER", "percentile")
        self.normalizer_params = self.get("NORMALIZER_PARAMS",
                                          "{'percentile': 95}")

        # Path to store all relevant data
        self.log_base = self.get("LOGS", "./logs")
        self.logs = os.path.join(self.log_base, "last")

    def get_character(self):
        return [self.model, self.instrumental, self.chopname,
                eval(self.chopparams).get('upper', False),
                self.loss, self.optimizer, self.normalizer]

    def create_logdir(self):
        self.logs = os.path.join(self.log_base, self.get_logname())
        if not os.path.exists(self.logs):
            os.makedirs(self.logs)
        # create symlink
        last_path = os.path.join(self.log_base, "last")
        if os.path.lexists(last_path):
            os.remove(last_path)
        os.symlink(os.path.abspath(self.logs), last_path)

    def get_logname(self):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return date + "_" + "-".join([str(x) for x in self.get_character()])

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
        # self is changing every time
        # calculate hash only over character
        val = md5(str(self.get_character()).encode()).hexdigest()
        return int(val, 16)


# create singleton config
config = Config()

if __name__ == "__main__":
    config = Config()
    print(config)
