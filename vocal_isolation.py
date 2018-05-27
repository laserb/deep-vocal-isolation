#!/usr/bin/python3
"""
VocalIsolation class

Main entry point of the application

Two main usages:

    python3 vocal_isolation.py
    => executes a training using the configurations in Config

    python3 vocal_isolation.py song.wav
    => isolates vocal from 'song.wav'
"""

import random
import string
import os
import sys
import signal

import numpy as np
from keras.utils import plot_model

import console
import conversion

from data import Data, remove_track_boundaries
from config import config
from metrics import Metrics
from checkpointer import Checkpointer
from modeler import Modeler
from loss import Loss
from optimizer import Optimizer
from chopper import Chopper
from normalizer import Normalizer
from batch import Batch


class VocalIsolation:
    def __init__(self, config):
        self.config = config
        metrics = Metrics().get()
        m = Modeler().get()
        loss = Loss().get()
        optimizer = Optimizer().get()
        console.log("Model has", m.count_params(), "params")
        m.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        m.summary(line_length=150)
        self.model = m
        # need to know so that we can avoid rounding errors with spectrogram
        # this should represent how much the input gets downscaled
        # in the middle of the network
        self.peakDownscaleFactor = 4

    def train(self, data, epochs, batch=8, start_epoch=0):
        x_train, y_train = data.train()
        x_valid, y_valid = data.valid()
        self.x_valid, self.y_valid = x_valid, y_valid
        checkpointer = Checkpointer(self)
        checkpoints = checkpointer.get()
        if self.config.batch_generator != "keras":
            batch_generator = Batch().get()
        if self.config.epoch_steps:
            epoch_steps = self.config.epoch_steps
        else:
            epoch_steps = remove_track_boundaries(x_train).shape[0]
        epoch_steps = epoch_steps // batch
        while epochs > 0:
            end_epoch = start_epoch + epochs
            console.log("Training for", epochs, "epochs on",
                        epoch_steps * batch, "examples")
            console.log("Validate on", len(x_valid), "examples")
            if self.config.batch_generator == "keras":
                x_train = remove_track_boundaries(x_train)
                y_train = remove_track_boundaries(y_train)
                history = self.model.fit(
                    x_train, y_train, batch_size=batch,
                    initial_epoch=start_epoch, epochs=end_epoch,
                    validation_data=(x_valid, y_valid),
                    callbacks=checkpoints)
            else:
                history = self.model.fit_generator(
                    batch_generator(x_train, y_train, batch_size=batch),
                    initial_epoch=start_epoch, epochs=end_epoch,
                    steps_per_epoch=epoch_steps,
                    validation_data=(x_valid, y_valid),
                    callbacks=checkpoints)
            console.notify(str(epochs) + " Epochs Complete!",
                           "Training on", data.in_path, "with size", batch)

            start_epoch += epochs
            if self.config.quit:
                break
            else:
                while True:
                    try:
                        epochs = int(
                            input("How many more epochs should we train for?"))
                        break
                    except ValueError:
                        console.warn(
                            "Oops, number parse failed. Try again, I guess?")
                if epochs > 0:
                    save = input("Should we save intermediate weights [y/n]? ")
                    if not save.lower().startswith("n"):
                        weight_path = ''.join(random.choice(string.digits)
                                              for _ in range(16)) + ".h5"
                        os.path.join(os.path.dirname(config.weights),
                                     weight_path)
                        console.log("Saving intermediate weights to",
                                    weight_path)
                        self.save_weights(weight_path)
        return history

    def run(self, data):
        self.config.create_logdir()
        # save current environment for later usage
        last_env = os.path.join(self.config.logs, "env")
        config_str = str(self.config)
        with open(last_env, "w") as f:
            f.write(config_str)

        plot_model(self.model, show_shapes=True,
                   to_file=os.path.join(self.config.logs, 'model.png'))

        history = self.train(data, self.config.epochs,
                             self.config.batch, self.config.start_epoch)

        self.save_weights(self.config.weights)
        metrics_path = os.path.join(self.config.logs, "metrics")
        with open(metrics_path, "w") as f:
            metric_names = list(history.history.keys())
            for name in metric_names:
                f.write("%s %s\n" % (name,
                                     history.history[name][-1]))
        return history

    def save_weights(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.config.logs, path)
        self.model.save_weights(path, overwrite=True)

    def load_weights(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.config.logs, path)
        self.model.load_weights(path)

    def process_spectrogram(self, spectrogram, channels=1):
        chopper = Chopper()
        chopper.name = "infer"
        chopper.params = "{'scale': %d}" % self.config.inference_slice
        chop = chopper.get(both=False)

        slices = chop(spectrogram)

        normalizer = Normalizer()
        normalize = normalizer.get(both=False)
        denormalize = normalizer.get_reverse()

        new_spectrogram = np.zeros((spectrogram.shape[0], 0, channels))
        for slice in slices:
            # normalize
            slice, norm = normalize(slice)

            epanded_spectrogram = conversion.expand_to_grid(
                slice, self.peakDownscaleFactor, channels)
            epanded_spectrogram_with_batch_and_channels = \
                epanded_spectrogram[np.newaxis, :, :]

            predicted_spectrogram_with_batch_and_channels = self.model.predict(
                epanded_spectrogram_with_batch_and_channels)
            # o /// o
            predicted_spectrogram = \
                predicted_spectrogram_with_batch_and_channels[0, :, :, :]
            local_spectrogram = predicted_spectrogram[:slice.shape[0],
                                                      :slice.shape[1], :]

            # de-normalize
            local_spectrogram = denormalize(local_spectrogram, norm)

            new_spectrogram = np.concatenate(
                    (new_spectrogram, local_spectrogram), axis=1)
        console.log("Processed spectrogram")
        return spectrogram, new_spectrogram

    def infer(self, path, fft_window_size, phase_iterations=10,
              learn_phase=False, channels=1):
        console.log("Attempting to isolate vocals from", path)
        audio, sample_rate = conversion.load_audio_file(path)
        spectrogram = conversion.audio_file_to_spectrogram(
            audio, fft_window_size=fft_window_size,
            learn_phase=self.config.learn_phase)
        console.log("Retrieved spectrogram; processing...")

        info = self.process_spectrogram(spectrogram, channels)
        spectrogram, new_spectrogram = info

        console.log("reconverting to audio")

        # save original spectrogram as image
        path_parts = os.path.split(path)
        filename_parts = os.path.splitext(path_parts[1])

        conversion.save_spectrogram(spectrogram, os.path.join(
            path_parts[0], filename_parts[0]) + ".png")

        # save network output
        self.save_audio(new_spectrogram,
                        fft_window_size,
                        phase_iterations,
                        sample_rate,
                        path,
                        vocal=not self.config.instrumental,
                        learn_phase=learn_phase)

        # save difference
        self.save_audio(spectrogram - new_spectrogram,
                        fft_window_size,
                        phase_iterations,
                        sample_rate,
                        path,
                        vocal=self.config.instrumental,
                        learn_phase=learn_phase)

        console.log("Vocal isolation complete")

    def save_audio(self, spectrogram, fft_window_size,
                   phase_iterations, sample_rate,
                   path, vocal=True, learn_phase=False):
        part = "_vocal" if vocal else "_instrumental"
        new_audio = conversion.spectrogram_to_audio_file(
                spectrogram,
                fft_window_size=fft_window_size,
                phase_iterations=phase_iterations,
                learn_phase=learn_phase)
        path_parts = os.path.split(path)
        filename_parts = os.path.splitext(path_parts[1])
        output_filename_base = os.path.join(
            path_parts[0], filename_parts[0] + part)
        console.log("Converted to audio; writing to",
                    output_filename_base + ".wav")

        conversion.save_audio_file(
            new_audio, output_filename_base + ".wav", sample_rate)
        conversion.save_spectrogram(spectrogram, output_filename_base + ".png")


def get_signal_handler(vocal_isolation):
    def signal_handler(signal, frame):
        save = input("Should we save intermediate weights [y/n]? ")
        if not save.lower().startswith("n"):
            vocal_isolation.save_weights(vocal_isolation.config.weights)
        sys.exit(0)
    return signal_handler


if __name__ == "__main__":
    files = sys.argv[1:]
    config_str = str(config)
    print(config_str)

    vocal_isolation = VocalIsolation(config)

    if len(files) == 0 and config.data:
        console.log("No files provided; attempting to train on " +
                    config.data + "...")
        if config.batch_generator.startswith("random") \
                and config.epoch_steps == 0:
            console.error("EPOCH_STEPS is not set,"
                          " but cannot be determined from data.")
            exit(1)
        if config.load:
            console.h1("Loading Weights")
            vocal_isolation.load_weights(config.weights)
        console.h1("Loading Data")
        data = Data()
        console.h1("Training Model")
        signal.signal(signal.SIGINT, get_signal_handler(vocal_isolation))
        vocal_isolation.run(data)
    elif len(files) > 0:
        console.log("Weights provided; performing inference on " +
                    str(files) + "...")
        console.h1("Loading weights")
        vocal_isolation.load_weights(config.weights)
        for f in files:
            vocal_isolation.infer(f, config.fft,
                                  config.phase_iterations,
                                  config.learn_phase,
                                  config.get_channels())
    else:
        console.error(
            "Please provide data to train on (--data) or files to infer on")
