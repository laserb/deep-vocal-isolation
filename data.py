"""
Loads and stores mashup data given a folder full of acapellas and instrumentals
Assumes that all audio clips (wav, mp3) in the folder
a) have "acapella" in the filename if they're an acapella
b) have "instrumental" in the filename if they're an instrumentalü
"""
import sys
import os
import numpy as np
import h5py

import console
import conversion
from config import config
from chopper import Chopper
from normalizer import Normalizer


def remove_track_boundaries(tracks):
    slices = []
    for track in tracks:
        slices.extend(track)
    return np.array(slices)


class Data:
    def __init__(self):
        self.config = config
        self.inPath = self.config.data
        self.fftWindowSize = self.config.fft
        self.trainingSplit = self.config.split
        self.isInstrumental = self.config.instrumental
        self.mashup = []
        self.acapella = []
        self.instrumental = []

        self.load()

    def train(self):
        length = len(self.mashup) * self.trainingSplit
        if self.config.batch_generator.startswith("random"):
            return self.prepare_random_data(end=length)
        else:
            chop = Chopper().get()
            return self.prepare_data(chop, end=length)

    def valid(self):
        chopper = Chopper()
        params = eval(chopper.params)
        chopper.name = "full"
        params["upper"] = False
        chopper.params = str(params)
        chop = chopper.get()
        start = len(self.mashup) * self.trainingSplit
        xValid, yValid = self.prepare_data(chop, start=start)
        xValid = remove_track_boundaries(xValid)
        yValid = remove_track_boundaries(yValid)
        return xValid, yValid

    def prepare_data(self, chop, start=0, end=None, post_process=False):
        normalize = Normalizer().get()

        if end is None:
            end = len(self.mashup)
        x = self.mashup[int(start): int(end)]
        if self.isInstrumental:
            y = self.instrumental[int(start): int(end)]
        else:
            y = self.acapella[int(start): int(end)]

        x = [self.prepare_spectrogram(s) for s in x]
        y = [self.prepare_spectrogram(s) for s in y]

        mashupSlices = []
        outputSlices = []
        for mashup, output in zip(x, y):
            xSlices, ySlices = chop(mashup, output)
            xSlices, ySlices = \
                normalize(xSlices, ySlices)

            xSlices = np.array(xSlices)[:]
            ySlices = np.array(ySlices)[:]
            mashupSlices.append(xSlices)
            outputSlices.append(ySlices)
        return mashupSlices, outputSlices

    def prepare_random_data(self, start=0, end=None, post_process=False):
        if end is None:
            end = len(self.mashup)
        x = self.mashup[int(start): int(end)]
        if self.isInstrumental:
            y = self.instrumental[int(start): int(end)]
        else:
            y = self.acapella[int(start): int(end)]

        x = [self.prepare_spectrogram(s) for s in x]
        y = [self.prepare_spectrogram(s) for s in y]

        return x, y

    def prepare_spectrogram(self, spectrogram):
        if self.config.learn_phase:
            return conversion.stftToRealAndImag(spectrogram)
        else:
            return conversion.stftToAmplitude(spectrogram)

    def get_data_path(self):
        return os.path.join(self.inPath, "data_%s.h5" % self.fftWindowSize)

    def load(self, saveDataAsH5=True):
        def checkFilename(f):
            return (f.endswith(".mp3") or f.endswith("_all.wav")) \
                and not f.startswith(".")

        h5Path = self.get_data_path()
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")
            mashup = h5f["mashup"]
            acapella = h5f["acapella"]
            instrumental = h5f["instrumental"]
            for track in sorted(mashup.keys()):
                self.mashup.append(mashup[track])
                self.acapella.append(acapella[track])
                self.instrumental.append(instrumental[track])
        else:
            for dirPath, dirNames, fileNames in os.walk(self.inPath):
                filteredFiles = filter(checkFilename, fileNames)
                for fileName in filteredFiles:
                    fileName = os.path.join(self.inPath, fileName)
                    acapella_file = fileName.replace("_all.wav",
                                                     "_acapella.wav")
                    instrumental_file = fileName.replace("_all.wav",
                                                         "_instrumental.wav")
                    if not all([os.path.exists(acapella_file),
                                os.path.exists(instrumental_file)]):
                        continue
                    audio, sampleRate = conversion.loadAudioFile(fileName)
                    spectrogram = conversion.audioFileToStft(
                        audio, self.fftWindowSize)
                    mashup = spectrogram

                    audio, sampleRate = conversion.loadAudioFile(acapella_file)
                    spectrogram = conversion.audioFileToStft(
                        audio, self.fftWindowSize)
                    acapella = spectrogram

                    audio, sampleRate = \
                        conversion.loadAudioFile(instrumental_file)
                    spectrogram = conversion.audioFileToStft(
                        audio, self.fftWindowSize)
                    instrumental = spectrogram

                    console.info("Created spectrogram for", fileName,
                                 "with shape",
                                 spectrogram.shape)
                    self.mashup.append(mashup)
                    self.acapella.append(acapella)
                    self.instrumental.append(instrumental)
            console.info("Created", len(self.mashup), "total spectras")
            # Save to file
            if saveDataAsH5:
                self.save()

    def save(self):
        h5Path = self.get_data_path()
        h5f = h5py.File(h5Path, "w")
        mashup = h5f.create_group("mashup")
        acapella = h5f.create_group("acapella")
        instrumental = h5f.create_group("instrumental")
        for i in range(len(self.mashup)):
            track = 'track{:d}'.format(i)
            mashup.create_dataset(name=track, data=self.mashup[i])
            acapella.create_dataset(name=track, data=self.acapella[i])
            instrumental.create_dataset(name=track, data=self.instrumental[i])
        h5f.close()


if __name__ == "__main__":
    # Simple testing code to use while developing
    console.h1("Loading Data")
    d = Data()
    console.h1("Writing Sample Data")
    if not os.path.exists("sample"):
        os.mkdir("sample")
    n = 5
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    for i in range(n):
        conversion.saveSpectrogram(d.x[i].squeeze(),
                                   "sample/x_sample_{}.png".format(i))
        conversion.saveSpectrogram(d.y[i].squeeze(),
                                   "sample/y_sample_{}.png".format(i))
        audio = conversion.spectrogramToAudioFile(d.x[i].squeeze(), 1536)
        conversion.saveAudioFile(audio,
                                 "sample/x_sample_{}.wav".format(i),
                                 22050)
