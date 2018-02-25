"""
Loads and stores mashup data given a folder full of acapellas and instrumentals
Assumes that all audio clips (wav, mp3) in the folder
a) have their Camelot key as the first token in the filename
b) are in the same BPM
c) have "acapella" somewhere in the filename if they're an acapella, and are otherwise instrumental  # noqa: E501
d) all have identical arrangements
e) have the same sample rate
"""
import sys
import os
import numpy as np
import h5py

import console
import conversion
from config import Config
from chopper import Chopper

# Modify these functions if your data is in a different format


def keyOfFile(fileName):
    firstToken = int(fileName.split()[0])
    if 0 < firstToken <= NUMBER_OF_KEYS:
        return firstToken
    console.warn("File", fileName, "doesn't specify its key, ignoring..")
    return None


def fileIsAcapella(fileName):
    return "acapella" in fileName.lower()


NUMBER_OF_KEYS = 12  # number of keys to iterate over


class Data:
    def __init__(self):
        self.config = Config()
        self.inPath = self.config.data
        self.fftWindowSize = self.config.fft
        self.trainingSplit = self.config.split
        self.instrumental = self.config.instrumental
        self.x = []
        self.y = []
        self.load()

    def train(self):
        return (self.x[:int(len(self.x) * self.trainingSplit)],
                self.y[:int(len(self.y) * self.trainingSplit)])

    def valid(self):
        return (self.x[int(len(self.x) * self.trainingSplit):],
                self.y[int(len(self.y) * self.trainingSplit):])

    def get_data_path(self, data_hash):
        if self.instrumental:
            type_string = "instrumental"
        else:
            type_string = "acapella"
        return os.path.join(self.inPath,
                            "data_%s_%s.h5" % (type_string, data_hash))

    def load(self, saveDataAsH5=True):
        def checkFilename(f):
            return (f.endswith(".mp3") or f.endswith("_all.wav")) \
                and not f.startswith(".")

        chopper = Chopper()
        data_hash = hash(chopper)
        print("Data hash: %s" % data_hash)
        h5Path = self.get_data_path(data_hash)
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")
            self.x = h5f["x"][:]
            self.y = h5f["y"][:]
        else:
            chop = chopper.get()
            for dirPath, dirNames, fileNames in os.walk(self.inPath):
                filteredFiles = filter(checkFilename, fileNames)
                for fileName in filteredFiles:
                    if self.instrumental:
                        acapella_file = fileName.replace("_all.wav",
                                                         "_instrumental.wav")
                    else:
                        acapella_file = fileName.replace("_all.wav",
                                                         "_acapella.wav")
                    if not os.path.exists(os.path.join(self.inPath,
                                                       acapella_file)):
                        continue
                    audio, sampleRate = conversion.loadAudioFile(
                        os.path.join(self.inPath, fileName))
                    spectrogram, phase = conversion.audioFileToSpectrogram(
                        audio, self.fftWindowSize)
                    mashup = spectrogram

                    audio, sampleRate = conversion.loadAudioFile(
                        os.path.join(self.inPath, acapella_file))
                    spectrogram, phase = conversion.audioFileToSpectrogram(
                        audio, self.fftWindowSize)
                    acapella = spectrogram

                    console.info("Created spectrogram for", fileName,
                                 "with shape",
                                 spectrogram.shape)
                    mashupSlices = chop(mashup)
                    acapellaSlices = chop(acapella)
                    self.x.extend(mashupSlices)
                    self.y.extend(acapellaSlices)
            console.info("Created", len(self.x), "total slices so far")
            # Add a "channels" channel to please the network
            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]
            # Save to file
            if saveDataAsH5:
                self.save(data_hash)

    def save(self, data_hash):
        h5Path = self.get_data_path(data_hash)
        h5f = h5py.File(h5Path, "w")
        h5f.create_dataset("x", data=self.x)
        h5f.create_dataset("y", data=self.y)
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
