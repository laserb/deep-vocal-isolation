#!/usr/bin/python3
import yaml
import csv
import sys
import os
import console
from config import config
from data import Data
from acapellabot import AcapellaBot


class MatrixRunner(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = config
        self.train_data = None
        self.outfile = os.path.join(self.config.log_base, "result.md")

    def read_config(self, path):
        with open(path, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

    def create_config(self, names, current={}):
        if not names:
            self.config.create_logdir()
            yield current
        else:
            name = names[0]
            for value in self.data[name]:
                setattr(self.config, name, value)
                current[name] = value
                yield from self.create_config(names[1:], current)

    def train(self, current_config):
        acapellabot = AcapellaBot(self.config)
        metrics = acapellabot.run(self.train_data)
        names = sorted(list(current_config.keys()))
        values = [current_config[name] for name in names]
        self.resultwriter.writerow(values + metrics)
        self.csvfile.flush()

    def run(self):
        self.data = self.read_config(self.config_path)
        combinations = 1
        for values in self.data.values():
            combinations *= len(values)
        console.warn("Running on ", combinations, " combinations.")

        with open(self.outfile, "w") as self.csvfile:
            self.resultwriter = csv.writer(self.csvfile, delimiter='|',
                                           quotechar='"',
                                           quoting=csv.QUOTE_MINIMAL)

            metric_names = ["loss"] + self.config.metrics.split(",")
            headers = sorted(list(self.data.keys())) + metric_names
            self.resultwriter.writerow(headers)
            lines = ["-"*len(head) for head in headers]
            self.resultwriter.writerow(lines)
            self.csvfile.flush()

            self.train_data = Data()
            for current_config in self.create_config(list(self.data.keys())):
                self.train(current_config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "matrix.yml"
    runner = MatrixRunner(path)
    runner.run()
