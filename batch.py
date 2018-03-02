#!/usr/bin/python3
import numpy as np
import random
from config import config
from data import remove_track_boundaries


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

    # Shuffle slices before each epoch.
    # Train with every slice for one epoch.
    def default(self):
        def generator(features, labels, batch_size):
            # remove track boundaries
            features = remove_track_boundaries(features)
            labels = remove_track_boundaries(labels)
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

    # Select a slice from a random track.
    # Repeat until batch is full.
    # Not every slice is used for training.
    def tracks(self):
        def generator(features, labels, batch_size):
            # Create empty arrays to contain batch of features and labels#
            shape = features[0][0].shape
            batch_features = np.zeros((batch_size, *shape))
            batch_labels = np.zeros((batch_size, *shape))
            nTracks = len(features)
            while True:
                for i in range(batch_size):
                    # get random track
                    t = random.randrange(nTracks)
                    xTrack = features[t]
                    yTrack = labels[t]

                    # get random slice
                    s = random.randrange(len(xTrack))

                    batch_features[i] = xTrack[s]
                    batch_labels[i] = yTrack[s]
                yield batch_features, batch_labels
        return generator
