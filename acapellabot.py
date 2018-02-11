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
import datetime

import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, \
        UpSampling2D, Concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

import console
import conversion
import keras.backend as K

from data import Data
from config import Config


class AcapellaBot:
    def __init__(self, config):
        self.config = config
        mashup = Input(shape=(None, None, 1), name='input')
        convA = Conv2D(64, 3, activation='relu', padding='same')(mashup)
        conv = Conv2D(64, 4, strides=2, activation='relu',
                      padding='same', use_bias=False)(convA)
        conv = BatchNormalization()(conv)

        convB = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 4, strides=2, activation='relu',
                      padding='same', use_bias=False)(convB)
        conv = BatchNormalization()(conv)

        conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(128, 3, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convB])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convA])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(1, 3, activation='relu', padding='same')(conv)
        acapella = conv
        m = Model(inputs=mashup, outputs=acapella)
        console.log("Model has", m.count_params(), "params")
        m.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=[self.mean_pred])
        m.summary(line_length=150)
        plot_model(m, to_file='model.png', show_shapes=True)
        self.model = m
        # need to know so that we can avoid rounding errors with spectrogram
        # this should represent how much the input gets downscaled
        # in the middle of the network
        self.peakDownscaleFactor = 4

    def mean_pred(self, y_true, y_pred):
        return K.mean(y_pred)

    def train(self, data, epochs, batch=8, start_epoch=0):
        xTrain, yTrain = data.train()
        xValid, yValid = data.valid()
        while epochs > 0:
            end_epoch = start_epoch + epochs
            console.log("Training for", epochs, "epochs on",
                        len(xTrain), "examples")
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            checkpointer = ModelCheckpoint(filepath='./weights.hdf5',
                                           verbose=1,
                                           save_best_only=True)
            tensor_board = TensorBoard(log_dir=self.config.logPath +
                                       "/{}".format(date))
            self.model.fit(xTrain, yTrain, batch_size=batch,
                           initial_epoch=start_epoch, epochs=end_epoch,
                           validation_data=(xValid, yValid),
                           callbacks=[checkpointer, tensor_board])
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
                        console.log("Saving intermediate weights to",
                                    weightPath)
                        self.saveWeights(weightPath)

    def saveWeights(self, path):
        self.model.save_weights(path, overwrite=True)

    def loadWeights(self, path):
        self.model.load_weights(path)

    def isolateVocals(self, path, fftWindowSize, phaseIterations=10):
        console.log("Attempting to isolate vocals from", path)
        audio, sampleRate = conversion.loadAudioFile(path)
        spectrogram, phase = conversion.audioFileToSpectrogram(
            audio, fftWindowSize=fftWindowSize)
        console.log("Retrieved spectrogram; processing...")

        expandedSpectrogram = conversion.expandToGrid(
            spectrogram, self.peakDownscaleFactor)
        expandedSpectrogramWithBatchAndChannels = \
            expandedSpectrogram[np.newaxis, :, :, np.newaxis]
        predictedSpectrogramWithBatchAndChannels = self.model.predict(
            expandedSpectrogramWithBatchAndChannels)
        # o /// o
        predictedSpectrogram = \
            predictedSpectrogramWithBatchAndChannels[0, :, :, 0]
        newSpectrogram = predictedSpectrogram[:spectrogram.shape[0],
                                              :spectrogram.shape[1]]
        console.log("Processed spectrogram; reconverting to audio")

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
                       vocal=not self.config.instrumental)

        # save difference
        self.saveAudio(spectrogram - newSpectrogram,
                       fftWindowSize,
                       phaseIterations,
                       sampleRate,
                       path,
                       vocal=self.config.instrumental)

        console.log("Vocal isolation complete")

    def saveAudio(self, spectrogram, fftWindowSize,
                  phaseIterations, sampleRate,
                  path, vocal=True):
        part = "_vocal" if vocal else "_instrumental"
        newAudio = conversion.spectrogramToAudioFile(
                spectrogram,
                fftWindowSize=fftWindowSize,
                phaseIterations=phaseIterations)
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])
        outputFileNameBase = os.path.join(
            pathParts[0], fileNameParts[0] + part)
        console.log("Converted to audio; writing to",
                    outputFileNameBase + ".wav")

        conversion.saveAudioFile(
            newAudio, outputFileNameBase + ".wav", sampleRate)
        conversion.saveSpectrogram(spectrogram, outputFileNameBase + ".png")


if __name__ == "__main__":
    files = sys.argv[1:]
    config = Config()
    config_str = str(config)
    print(config_str)
    # save current environment for later usage
    with open("./envs/last", "w") as f:
        f.write(config_str)

    acapellabot = AcapellaBot(config)

    if len(files) == 0 and config.data:
        console.log("No files provided; attempting to train on " +
                    config.data + "...")
        if config.load:
            console.h1("Loading Weights")
            acapellabot.loadWeights(config.weights)
        console.h1("Loading Data")
        data = Data()
        console.h1("Training Model")
        acapellabot.train(data, config.epochs,
                          config.batch, config.start_epoch)
        acapellabot.saveWeights(config.weights)
    elif len(files) > 0:
        console.log("Weights provided; performing inference on " +
                    str(files) + "...")
        console.h1("Loading weights")
        acapellabot.loadWeights(config.weights)
        for f in files:
            acapellabot.isolateVocals(f, config.fft, config.phase)
    else:
        console.error(
            "Please provide data to train on (--data) or files to infer on")
