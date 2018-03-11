"""
Acapella extraction with a CNN

Typical usage:
    python acapellabot.py song.wav
    => Extracts acapella from <song.wav>
       to <song (Acapella Attempt).wav> using default weights

    python acapellabot.py --data input_folder \
            --batch 32 --weights new_model_iteration.h5
    => Trains a new model based on song/acapella pairs
       in the folder <input_folder>
       and saves weights to <new_model_iteration.h5> once complete.
       See data.py for data specifications.
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


class AcapellaBot:
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
        xTrain, yTrain = data.train()
        xValid, yValid = data.valid()
        self.xValid, self.yValid = xValid, yValid
        checkpointer = Checkpointer(self)
        checkpoints = checkpointer.get()
        if self.config.batch_generator != "keras":
            batch_generator = Batch().get()
        if self.config.epoch_steps:
            epoch_steps = self.config.epoch_steps
        else:
            epoch_steps = remove_track_boundaries(xTrain).shape[0]
        epoch_steps = epoch_steps // batch
        while epochs > 0:
            end_epoch = start_epoch + epochs
            console.log("Training for", epochs, "epochs on",
                        epoch_steps * batch, "examples")
            console.log("Validate on", len(xValid), "examples")
            if self.config.batch_generator == "keras":
                xTrain = remove_track_boundaries(xTrain)
                yTrain = remove_track_boundaries(yTrain)
                history = self.model.fit(
                    xTrain, yTrain, batch_size=batch,
                    initial_epoch=start_epoch, epochs=end_epoch,
                    validation_data=(xValid, yValid),
                    callbacks=checkpoints)
            else:
                history = self.model.fit_generator(
                    batch_generator(xTrain, yTrain, batch_size=batch),
                    initial_epoch=start_epoch, epochs=end_epoch,
                    steps_per_epoch=epoch_steps,
                    validation_data=(xValid, yValid),
                    callbacks=checkpoints)
            console.notify(str(epochs) + " Epochs Complete!",
                           "Training on", data.inPath, "with size", batch)

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
                        weightPath = ''.join(random.choice(string.digits)
                                             for _ in range(16)) + ".h5"
                        os.path.join(os.path.dirname(config.weights),
                                     weightPath)
                        console.log("Saving intermediate weights to",
                                    weightPath)
                        self.saveWeights(weightPath)
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

        self.saveWeights(self.config.weights)
        metrics_path = os.path.join(self.config.logs, "metrics")
        with open(metrics_path, "w") as f:
            metric_names = list(history.history.keys())
            for name in metric_names:
                f.write("%s %s\n" % (name,
                                     history.history[name][-1]))
        return history

    def saveWeights(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.config.logs, path)
        self.model.save_weights(path, overwrite=True)

    def loadWeights(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.config.logs, path)
        self.model.load_weights(path)

    def process_spectrogram(self, spectrogram, channels=1):
        chopper = Chopper()
        chopper.name = "infere"
        chopper.params = "{'scale': %d}" % self.config.inference_slice
        chop = chopper.get(both=False)

        slices = chop(spectrogram)

        newSpectrogram = np.zeros((spectrogram.shape[0], 0, channels))
        for slice in slices:
            expandedSpectrogram = conversion.expandToGrid(
                slice, self.peakDownscaleFactor, channels)
            expandedSpectrogramWithBatchAndChannels = \
                expandedSpectrogram[np.newaxis, :, :]

            predictedSpectrogramWithBatchAndChannels = self.model.predict(
                expandedSpectrogramWithBatchAndChannels)
            # o /// o
            predictedSpectrogram = \
                predictedSpectrogramWithBatchAndChannels[0, :, :, :]
            localSpectrogram = predictedSpectrogram[:slice.shape[0],
                                                    :slice.shape[1], :]

            newSpectrogram = np.concatenate((newSpectrogram, localSpectrogram),
                                            axis=1)
        console.log("Processed spectrogram")
        return spectrogram, newSpectrogram

    def isolateVocals(self, path, fftWindowSize, phaseIterations=10,
                      learnPhase=False, channels=1):
        console.log("Attempting to isolate vocals from", path)
        audio, sampleRate = conversion.loadAudioFile(path)
        spectrogram = conversion.audioFileToSpectrogram(
            audio, fftWindowSize=fftWindowSize,
            learn_phase=self.config.learn_phase)
        console.log("Retrieved spectrogram; processing...")

        normalizer = Normalizer()
        normalize = normalizer.get(both=False)
        denormalize = normalizer.get_reverse()

        # normalize
        spectogram, norm = normalize(spectrogram)

        info = self.process_spectrogram(spectrogram, channels)
        spectrogram, newSpectrogram = info

        # de-normalize
        newSpectrogram = denormalize(newSpectrogram, norm)
        spectrogram = denormalize(spectrogram, norm)

        console.log("reconverting to audio")

        # save original spectrogram as image
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])

        conversion.saveSpectrogram(spectrogram, os.path.join(
            pathParts[0], fileNameParts[0]) + ".png")

        # save network output
        self.saveAudio(newSpectrogram,
                       fftWindowSize,
                       phaseIterations,
                       sampleRate,
                       path,
                       vocal=not self.config.instrumental,
                       learnPhase=learnPhase)

        # save difference
        self.saveAudio(spectrogram - newSpectrogram,
                       fftWindowSize,
                       phaseIterations,
                       sampleRate,
                       path,
                       vocal=self.config.instrumental,
                       learnPhase=learnPhase)

        console.log("Vocal isolation complete")

    def saveAudio(self, spectrogram, fftWindowSize,
                  phaseIterations, sampleRate,
                  path, vocal=True, learnPhase=False):
        part = "_vocal" if vocal else "_instrumental"
        newAudio = conversion.spectrogramToAudioFile(
                spectrogram,
                fftWindowSize=fftWindowSize,
                phaseIterations=phaseIterations,
                learnPhase=learnPhase)
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])
        outputFileNameBase = os.path.join(
            pathParts[0], fileNameParts[0] + part)
        console.log("Converted to audio; writing to",
                    outputFileNameBase + ".wav")

        conversion.saveAudioFile(
            newAudio, outputFileNameBase + ".wav", sampleRate)
        conversion.saveSpectrogram(spectrogram, outputFileNameBase + ".png")


def get_signal_handler(acapellabot):
    def signal_handler(signal, frame):
        save = input("Should we save intermediate weights [y/n]? ")
        if not save.lower().startswith("n"):
            acapellabot.saveWeights(acapellabot.config.weights)
        sys.exit(0)
    return signal_handler


if __name__ == "__main__":
    files = sys.argv[1:]
    config_str = str(config)
    print(config_str)

    acapellabot = AcapellaBot(config)

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
            acapellabot.loadWeights(config.weights)
        console.h1("Loading Data")
        data = Data()
        console.h1("Training Model")
        signal.signal(signal.SIGINT, get_signal_handler(acapellabot))
        acapellabot.run(data)
    elif len(files) > 0:
        console.log("Weights provided; performing inference on " +
                    str(files) + "...")
        console.h1("Loading weights")
        acapellabot.loadWeights(config.weights)
        for f in files:
            acapellabot.isolateVocals(f, config.fft,
                                      config.phase_iterations,
                                      config.learn_phase,
                                      config.get_channels())
    else:
        console.error(
            "Please provide data to train on (--data) or files to infer on")
