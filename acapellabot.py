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

import argparse
import random
import string
import os
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


class AcapellaBot:
    def __init__(self):
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
            tensor_board = TensorBoard(log_dir="./logs/{}".format(date))
            self.model.fit(xTrain, yTrain, batch_size=batch,
                           initial_epoch=start_epoch, epochs=end_epoch,
                           validation_data=(xValid, yValid),
                           callbacks=[checkpointer, tensor_board])
            console.notify(str(epochs) + " Epochs Complete!",
                           "Training on", data.inPath, "with size", batch)
            start_epoch += epochs
            while True:
                try:
                    epochs = int(
                        input("How many more epochs should we train for? "))
                    break
                except ValueError:
                    console.warn(
                        "Oops, number parse failed. Try again, I guess?")
            if epochs > 0:
                save = input("Should we save intermediate weights [y/n]? ")
                if not save.lower().startswith("n"):
                    weightPath = ''.join(random.choice(string.digits)
                                         for _ in range(16)) + ".h5"
                    console.log("Saving intermediate weights to", weightPath)
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
                       vocal=True)

        # save difference
        self.saveAudio(spectrogram - newSpectrogram,
                       fftWindowSize,
                       phaseIterations,
                       sampleRate,
                       path,
                       vocal=False)

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
    # if data folder is specified, create a new data object
    #    and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(
        description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int,
                        help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str,
                        help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float,
                        help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs to train.")
    parser.add_argument("--start-epoch", default=0, type=int,
                        help="First epoch number.")
    parser.add_argument("--weights", default="weights.h5",
                        type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int,
                        help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int,
                        help="Phase iterations for reconstruction")
    parser.add_argument("--load", action='store_true',
                        help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    acapellabot = AcapellaBot()

    if len(args.files) == 0 and args.data:
        console.log("No files provided; attempting to train on " +
                    args.data + "...")
        if args.load:
            console.h1("Loading Weights")
            acapellabot.loadWeights(args.weights)
        console.h1("Loading Data")
        data = Data(args.data, args.fft, args.split)
        console.h1("Training Model")
        acapellabot.train(data, args.epochs, args.batch, args.start_epoch)
        acapellabot.saveWeights(args.weights)
    elif len(args.files) > 0:
        console.log("Weights provided; performing inference on " +
                    str(args.files) + "...")
        console.h1("Loading weights")
        acapellabot.loadWeights(args.weights)
        for f in args.files:
            acapellabot.isolateVocals(f, args.fft, args.phase)
    else:
        console.error(
            "Please provide data to train on (--data) or files to infer on")
