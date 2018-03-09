#!/usr/bin/python3
import yaml
import csv
import sys
import os
import matplotlib
# The default tk backend does not work without X server
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import console  # noqa: E402
from config import config  # noqa: E402
from data import Data  # noqa: E402
from acapellabot import AcapellaBot  # noqa: E402


class MatrixRunner(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = config
        self.train_data = None
        self.outfile = os.path.join(self.config.log_base, "result.md")
        self.repeat = None
        self.ix = 0
        self.ids = []

    def read_config(self, path):
        with open(path, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

    def create_config(self, names, current={}):
        if not names:
            self.ix += 1
            if self.repeat is not None:
                for i in range(self.repeat):
                    current["ix"] = "%s-%s" % (self.ix, i)
                    self.config.create_logdir()
                    yield current
            else:
                current["ix"] = self.ix
                self.config.create_logdir()
                yield current
        else:
            name = names[0]
            for value in self.data[name]:
                setattr(self.config, name, value)
                current[name] = value
                yield from self.create_config(names[1:], current)

    def train(self, current_config):
        ix = current_config.pop("ix")
        self.ids.append(ix)
        acapellabot = AcapellaBot(self.config)
        history = acapellabot.run(self.train_data)
        metrics = [history.history["val_" + name][-1]
                   for name in self.metric_names]
        self.plt_loss.plot(history.history['loss'])
        self.plt_val_loss.plot(history.history['val_loss'])
        min_loss = min(history.history["val_loss"])
        names = sorted(list(current_config.keys()))
        values = [current_config[name] for name in names]
        values = [ix] + values
        self.resultwriter.writerow(values + [min_loss] + metrics)
        self.csvfile.flush()

    def run(self):
        f, axarr = plt.subplots(2)
        self.plt_loss, self.plt_val_loss = axarr

        self.data = self.read_config(self.config_path)
        self.repeat = self.data.pop("repeat", None)
        combinations = 1
        for values in self.data.values():
            combinations *= len(values)
        console.warn("Running on ", combinations, " combinations.")

        with open(self.outfile, "w") as self.csvfile:
            self.resultwriter = csv.writer(self.csvfile, delimiter='|',
                                           quotechar='"',
                                           quoting=csv.QUOTE_MINIMAL)

            self.metric_names = ["loss"] + self.config.metrics.split(",")
            headers = sorted(list(self.data.keys()))
            headers += ["min_loss"] + self.metric_names
            headers = ["id"] + headers
            self.resultwriter.writerow(headers)
            lines = ["-"*len(head) for head in headers]
            self.resultwriter.writerow(lines)
            self.csvfile.flush()

            self.train_data = Data()
            for current_config in self.create_config(list(self.data.keys())):
                self.train(current_config)

        self.plt_loss.set_title("loss")
        self.plt_val_loss.set_title("val_loss")

        self.plt_loss.legend(self.ids, loc='upper right')
        self.plt_val_loss.legend(self.ids, loc='upper right')
        plt.savefig(os.path.join(self.config.log_base, 'benchmark-loss.png'))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "matrix.yml"
    runner = MatrixRunner(path)
    runner.run()
