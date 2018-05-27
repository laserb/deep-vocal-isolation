#!/usr/bin/python3
"""
Data class

Utility class for data preparation

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
        self.in_path = self.config.data
        self.fft_window_size = self.config.fft
        self.training_split = self.config.split
        self.is_instrumental = self.config.instrumental
        self.mashup = {}
        self.acapella = {}
        self.instrumental = {}
        self.track_names = []

        self.load()
        self.split_tracks()

    def split_tracks(self):
        self.validation_tracks = self.config.get_validation_tracks()
        self.test_tracks = self.config.get_test_tracks()

        self.validation_tracks = []
        for track in self.config.get_validation_tracks():
            for track_name in self.track_names:
                if track in track_name:
                    self.validation_tracks.append(track_name)

        self.test_tracks = []
        for track in self.config.get_test_tracks():
            for track_name in self.track_names:
                if track in track_name:
                    self.test_tracks.append(track_name)

        self.train_tracks = []
        for track in self.track_names:
            if track not in self.validation_tracks \
                    and track not in self.test_tracks:
                self.train_tracks.append(track)

        if not self.validation_tracks:
            length = int(len(self.track_names) * self.training_split)
            self.validation_tracks = self.track_names[length:]
            self.train_tracks = self.track_names[:length]

    def train(self):
        if self.config.batch_generator.startswith("random"):
            return self.prepare_random_data(self.train_tracks)
        else:
            chop = Chopper().get()
            return self.prepare_data(chop, self.train_tracks)

    def valid(self):
        chopper = Chopper()
        params = eval(chopper.params)
        params["upper"] = False
        chopper.name = "sliding_full"
        params["step"] = 32
        params["scale"] = 128
        chopper.params = str(params)
        chop = chopper.get()
        x_valid, y_valid = self.prepare_data(chop, self.validation_tracks)
        x_valid = remove_track_boundaries(x_valid)
        y_valid = remove_track_boundaries(y_valid)
        return x_valid, y_valid

    def prepare_data(self, chop, tracks, post_process=False):
        normalize = Normalizer().get()

        x = []
        y = []
        for track in tracks:
            x.append(self.mashup[track])
            if self.is_instrumental:
                y.append(self.instrumental[track])
            else:
                y.append(self.acapella[track])

        x = [self.prepare_spectrogram(s) for s in x]
        y = [self.prepare_spectrogram(s) for s in y]

        x, y = normalize(x, y)

        mashup_slices = []
        output_slices = []
        for mashup, output in zip(x, y):
            x_slices, y_slices = chop(mashup, output)

            x_slices = np.array(x_slices)[:]
            y_slices = np.array(y_slices)[:]
            mashup_slices.append(x_slices)
            output_slices.append(y_slices)
        return mashup_slices, output_slices

    def prepare_random_data(self, tracks, post_process=False):
        normalize = Normalizer().get()
        x = []
        y = []
        for track in tracks:
            x.append(self.mashup[track])
            if self.is_instrumental:
                y.append(self.instrumental[track])
            else:
                y.append(self.acapella[track])

        x = [self.prepare_spectrogram(s) for s in x]
        y = [self.prepare_spectrogram(s) for s in y]
        x, y = normalize(x, y)
        return x, y

    def prepare_spectrogram(self, spectrogram):
        if self.config.learn_phase:
            return conversion.stft_to_real_and_imag(spectrogram)
        else:
            return conversion.stft_to_amplitude(spectrogram)

    def get_data_path(self):
        return os.path.join(self.in_path, "data_%s.h5" % self.fft_window_size)

    def load(self, save_data_as_h5=True):
        def check_filename(f):
            return (f.endswith(".mp3") or f.endswith("_all.wav")) \
                and not f.startswith(".")

        h5_path = self.get_data_path()
        if os.path.isfile(h5_path):
            h5f = h5py.File(h5_path, "r")
            mashup = h5f["mashup"]
            acapella = h5f["acapella"]
            instrumental = h5f["instrumental"]
            self.track_names = [name.decode("utf8")
                                for name in h5f["names"]["track"]]
            self.mashup = dict(mashup)
            self.acapella = dict(acapella)
            self.instrumental = dict(instrumental)
        else:
            for dirPath, dirNames, file_names in os.walk(self.in_path):
                filtered_files = filter(check_filename, file_names)
                for file_name in filtered_files:
                    name = file_name.replace("_all.wav", "")
                    file_name = os.path.join(self.in_path, file_name)
                    acapella_file = file_name.replace("_all.wav",
                                                      "_acapella.wav")
                    instrumental_file = file_name.replace("_all.wav",
                                                          "_instrumental.wav")
                    if not all([os.path.exists(acapella_file),
                                os.path.exists(instrumental_file)]):
                        continue

                    audio, sample_rate = conversion.load_audio_file(file_name)
                    spectrogram = conversion.audio_file_to_stft(
                        audio, self.fft_window_size)
                    mashup = spectrogram

                    audio, sample_rate = \
                        conversion.load_audio_file(acapella_file)
                    spectrogram = conversion.audio_file_to_stft(
                        audio, self.fft_window_size)
                    acapella = spectrogram

                    audio, sample_rate = \
                        conversion.load_audio_file(instrumental_file)
                    spectrogram = conversion.audio_file_to_stft(
                        audio, self.fft_window_size)
                    instrumental = spectrogram

                    console.info("Created spectrogram for", file_name,
                                 "with shape",
                                 spectrogram.shape)
                    self.mashup[name] = mashup
                    self.acapella[name] = acapella
                    self.instrumental[name] = instrumental
                    self.track_names.append(name)
            console.info("Created", len(self.mashup), "total spectras")
            # Save to file
            if save_data_as_h5:
                self.save()

    def save(self):
        h5_path = self.get_data_path()
        h5f = h5py.File(h5_path, "w")
        mashup = h5f.create_group("mashup")
        acapella = h5f.create_group("acapella")
        instrumental = h5f.create_group("instrumental")
        names = h5f.create_group("names")
        track_names = [name.encode("utf8") for name in self.track_names]
        names.create_dataset(name="track", data=track_names)
        for track in self.track_names:
            mashup.create_dataset(name=track.encode("utf8"),
                                  data=self.mashup[track])
            acapella.create_dataset(name=track.encode("utf8"),
                                    data=self.acapella[track])
            instrumental.create_dataset(name=track.encode("utf8"),
                                        data=self.instrumental[track])
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
        conversion.save_spectrogram(d.x[i].squeeze(),
                                    "sample/x_sample_{}.png".format(i))
        conversion.save_spectrogram(d.y[i].squeeze(),
                                    "sample/y_sample_{}.png".format(i))
        audio = conversion.spectrogram_to_audio_file(d.x[i].squeeze(), 1536)
        conversion.save_audio_file(audio,
                                   "sample/x_sample_{}.wav".format(i),
                                   22050)
