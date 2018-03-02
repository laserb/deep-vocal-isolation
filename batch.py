#!/usr/bin/python3
import numpy as np
import random
from config import config


class Batch(object):
    def __init__(self):
        self.config = config
        self.batch_generator = self.config.batch_generator

    def get(self):
        return getattr(self, self.batch_generator)()

    # If keras is set as batch generator
    # then the default keras functionality is used
    # and not this batch generator function.
    def keras(self):
        pass

    def default(self):
        def generator(features, labels, batch_size):
            # Create empty arrays to contain batch of features and labels#
            shape = features[0].shape
            batch_features = np.zeros((batch_size, *shape))
            batch_labels = np.zeros((batch_size, *shape))
            n = len(features)
            index = list(range(n))
            random.shuffle(index)
            j = 0
            while True:
                batch_index = index[j*batch_size:(j+1)*batch_size]
                j += 1
                # reset to satisfy queue generation
                if (j + 1) * batch_size > n:
                    j = 0
                for i in range(batch_size):
                    batch_features[i] = features[batch_index[i]]
                    batch_labels[i] = labels[batch_index[i]]
                yield batch_features, batch_labels
        return generator
