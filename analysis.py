#!/usr/bin/python3
import argparse
import os
import numpy as np
from config import config
import h5py

import matplotlib
# The default tk backend does not work without X server
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import conversion  # noqa: E402
from acapellabot import AcapellaBot  # noqa: E402
from data import Data  # noqa: E402
from normalizer import Normalizer  # noqa: E402


class Analysis:
    def __init__(self):
        self.config = config
        self.analyse = "spectrograms"
        self.save = True
        self.analysisPath = self.config.analysis_path
        self.content = "Analyse {} \n"

    def get(self):
        return getattr(self, self.analyse)

    def run(self, analyse, save, args):
        self.analyse = analyse
        self.save = save
        self.content = self.content.format(self.analyse)
        config_str = str(self.config)
        print(config_str)

        print(self.content)
        analyse = self.get()
        analyse(*args)

    def percentile(self):
        data = Data()

        self._do_percentile(data, data.mashup, "Mashup")
        self._do_percentile(data, data.instrumental, "Instrumental")
        self._do_percentile(data, data.acapella, "Acapella")

    def _do_percentile(self, data, spectrograms, name):

        pbar = None
        try:
            from progressbar import ProgressBar, Percentage, Bar
            pbar = ProgressBar(widgets=[Percentage(), Bar()],
                               maxval=len(data.track_names)*101)
            pbar.start()
        except Exception as e:
            pass

        k = 0
        if config.learn_phase:
            y_real = [[] for _ in range(101)]
            y_imag = [[] for _ in range(101)]
            for track in sorted(data.track_names):
                t = data.prepare_spectrogram(spectrograms[track])
                median_real = np.median(t[:, :, 0])
                median_imag = np.median(t[:, :, 1])
                for i in range(101):
                    if pbar is not None:
                        pbar.update(k)
                    k += 1
                    v = np.percentile(t[:, :, 0], i)
                    y_real[i].append(v-median_real)

                    v = np.percentile(t[:, :, 1], i)
                    y_imag[i].append(v-median_imag)

            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            h5f_path = os.path.join(self.analysisPath,
                                    "ir_percentile_%s.hdf5" % name)
            h5f = h5py.File(h5f_path, "w")
            h5f.create_dataset(name="real",
                               data=y_real)
            h5f.create_dataset(name="imag",
                               data=y_imag)
            h5f.close()

            plt.figure(figsize=(15, 15))
            plt.subplot(211)
            result = plt.boxplot(y_real, labels=range(101))
            print([l.get_ydata()[0] for l in result["medians"]])
            plt.xticks(rotation=90)
            plt.title("Real")
            plt.xlabel("percentile")
            plt.ylabel("difference from median")

            plt.subplot(212)
            result = plt.boxplot(y_imag, labels=range(101))
            print([l.get_ydata()[0] for l in result["medians"]])
            plt.xticks(rotation=90)
            plt.title("Imag")
            plt.xlabel("percentile")
            plt.ylabel("difference from median")
            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            plt.savefig(os.path.join(self.analysisPath,
                                     "percentile_%s_ir.png" % name))
            plt.close()
        else:
            y = [[] for _ in range(101)]
            for track in data.track_names:
                t = data.prepare_spectrogram(spectrograms[track])
                median = np.median(t)
                for i in range(101):
                    if pbar is not None:
                        pbar.update(k)
                    k += 1
                    v = np.percentile(t, i)
                    y[i].append(v-median)

            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            h5f_path = os.path.join(self.analysisPath,
                                    "amp_percentile_%s.hdf5" % name)
            h5f = h5py.File(h5f_path, "w")
            h5f.create_dataset(name="value",
                               data=y)
            h5f.close()

            plt.figure(figsize=(15, 15))
            result = plt.boxplot(y, labels=range(101))
            print([l.get_ydata()[0] for l in result["medians"]])
            plt.xticks(rotation=90)
            plt.title("Amplitude")
            plt.xlabel("percentile")
            plt.ylabel("difference from median")
            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            plt.savefig(os.path.join(self.analysisPath,
                                     "percentile_%s_amplitude.png" % name))
            plt.close()

    def stoi(self, filepath, clean_filepath=None):
        # filepath = path to mashup
        # Needs octave and octave-signal installed
        # Use "pip install oct2py" to install python - octave bridge
        # STOI assumes
        # * a sampling rate of 10kHz, resamples otherwise
        # * window length of 384ms
        # * 15 third octave bands over full frequency range
        # * overlapping segments with hanning window
        # * removes silent frames
        import librosa
        from oct2py import octave
        if clean_filepath is None:
            # No clean file given.
            # Get processed and clean file from mashup.
            acapellabot = AcapellaBot(config)
            acapellabot.loadWeights(config.weights)
            audio, sampleRate = conversion.load_audio_file(filepath)
            spectrogram = conversion.audio_file_to_spectrogram(
                audio, fftWindowSize=config.fft,
                learn_phase=self.config.learn_phase)

            normalizer = Normalizer()
            normalize = normalizer.get(both=False)
            denormalize = normalizer.get_reverse()

            # normalize
            spectogram, norm = normalize(spectrogram)

            info = acapellabot.process_spectrogram(spectrogram,
                                                   config.get_channels())
            spectrogram, new_spectrogram = info
            # de-normalize
            new_spectrogram = denormalize(new_spectrogram, norm)

            processed = conversion.spectrogram_to_audio_file(new_spectrogram,
                                                             config.fft,
                                                             config.phase)

            clean_filepath = filepath.replace("_all.wav", "_acapella.wav")
            clean, sampling_rate = librosa.load(clean_filepath)
        else:
            # A clean file is given.
            # Compare it with the processed audio.
            processed, sampling_rate = librosa.load(filepath)
            clean, sampling_rate = librosa.load(clean_filepath)

        # Make sure the original and processed audio have the same length
        clean = clean[:processed.shape[0]]

        octave.eval("pkg load signal")
        d = octave.stoi(clean, processed, sampling_rate)
        self._write("stoi: %f" % d)

    def mse(self, processed=None, vocal=None):
        self.mean_squared_error(processed, vocal)

    def mean_squared_error(self, processed_file=None, vocal_file=None):
        normalizer = Normalizer()
        normalize = normalizer.get(both=False)
        if processed_file is None:
            acapellabot = AcapellaBot(config)
            acapellabot.loadWeights(config.weights)
            data = Data()
            mses = []
            for track in data.validation_tracks + data.test_tracks:
                mashup = data.prepare_spectrogram(data.mashup[track])
                vocal = data.prepare_spectrogram(data.acapella[track])
                mashup, norm = normalize(mashup)
                vocal, _ = normalize(vocal, norm)
                info = acapellabot.process_spectrogram(mashup,
                                                       config.get_channels())
                new_spectrogram = info[1]
                mse = ((new_spectrogram - vocal)**2).mean()
                mses.append(mse)
                print(track, mse)
            print(np.mean(mses))
        else:
            vocal_audio, _ = conversion.load_audio_file(vocal_file)
            processed_audio, _ = conversion.load_audio_file(processed_file)

            # make sure audios have the same length
            vocal_audio = vocal_audio[:processed_audio.shape[0]]
            processed_audio = processed_audio[:vocal_audio.shape[0]]

            wave_mse = ((vocal_audio - processed_audio)**2).mean()

            print("\n")
            self._write("Wave mean squared error: %s" % wave_mse)

    def volume(self, filepath):
        normalizer = Normalizer()
        normalize = normalizer.get(both=False)
        denormalize = normalizer.get_reverse()

        vocal_file = filepath.replace("_all.wav", "_acapella.wav")
        instrumental_file = filepath.replace("_all.wav", "_instrumental.wav")

        acapellabot = AcapellaBot(config)
        acapellabot.loadWeights(config.weights)

        instrumental_audio, _ = conversion.load_audio_file(instrumental_file)
        vocal_audio, _ = conversion.load_audio_file(vocal_file)

        instrumental = conversion.audio_file_to_spectrogram(
            instrumental_audio, fftWindowSize=config.fft,
            learn_phase=self.config.learn_phase)
        vocal = conversion.audio_file_to_spectrogram(
            vocal_audio, fftWindowSize=config.fft,
            learn_phase=self.config.learn_phase)

        if not os.path.exists(self.analysisPath):
            os.mkdir(self.analysisPath)
        h5f_path = os.path.join(self.analysisPath,
                                "volume.hdf5")
        h5file = h5py.File(h5f_path, "w")

        ratio = 100
        x = [i/ratio for i in range(1, ratio)] + \
            [1] + \
            [ratio/i for i in range(ratio-1, 0, -1)]
        h5file.create_dataset(name="x", data=x)

        print("Unscaled original mix")
        mashup, norm = normalize(instrumental + vocal)
        acapella, _ = normalize(vocal, norm)
        info = acapellabot.process_spectrogram(mashup,
                                               config.get_channels())
        new_spectrogram = denormalize(info[1], norm)
        mse = ((new_spectrogram - vocal)**2).mean()
        y = [mse for _ in x]
        plt.loglog(x, y, label="baseline")
        h5file.create_dataset(name="baseline", data=y)

        original_ratio = np.max(vocal)/np.max(instrumental)
        print("Original ratio: %s" % original_ratio)
        vocal /= original_ratio

        print("Change vocal volume")
        y = []
        for i in x:
            mashup, norm = normalize(instrumental + i * vocal)
            acapella, _ = normalize(i * vocal, norm)
            info = acapellabot.process_spectrogram(mashup,
                                                   config.get_channels())
            new_spectrogram = denormalize(info[1], norm)
            if i != 0:
                new_spectrogram = new_spectrogram / i

            mse = ((new_spectrogram - vocal)**2).mean()
            y.append(mse)
            print(mse)
        plt.loglog(x, y, label="scaled")

        plt.xlabel("vocal/instrumental")
        plt.ylabel("mean squared error")
        plt.legend()

        h5file.create_dataset(name="scale", data=y)
        h5file.close()
        if not os.path.exists(self.analysisPath):
            os.mkdir(self.analysisPath)
        plt.savefig(os.path.join(self.analysisPath, "volume.png"))

    def distribution(self):
        data = Data()

        self._do_distribution(data, data.mashup, "Mashup")
        self._do_distribution(data, data.instrumental, "Instrumental")
        self._do_distribution(data, data.acapella, "Acapella")

    def _do_distribution_plot(self, pbar, h5f, data, spectrograms,
                              bin_range, part, prefix=""):
        k = 0
        vals = []
        for track in sorted(data.track_names):
            spectrogram = data.prepare_spectrogram(spectrograms[track])
            if pbar is not None:
                pbar.update(k)
            k += 1

            channel = 0
            if prefix == "imag":
                channel = 1

            window = spectrogram.shape[0] // 2
            if part == "upper":
                window_values = spectrogram[0:window, :, channel]
            elif part == "center":
                window_values = spectrogram[window // 2: window // 2 + window, :, channel]  # noqa
            else:
                window_values = spectrogram[-window:, :, channel]
            vals += window_values[:, :].flatten().tolist()

        if bin_range is None:
            if config.learn_phase:
                bin_min = np.percentile(vals, 1)
            else:
                bin_min = 0
            bin_max = np.percentile(vals, 99)
            bin_range = (bin_min, bin_max)

        values, bins, patches = plt.hist(vals,
                                         range=bin_range,
                                         bins=25,
                                         label="%s %s" % (prefix, part))
        plt.legend()
        h5f.create_dataset(name="%s_values" % part, data=values)
        h5f.create_dataset(name="%s_bins" % part, data=bins)
        del vals
        return bin_range

    def _do_distribution(self, data, spectrograms, name):

        pbar = None
        try:
            from progressbar import ProgressBar, Percentage, Bar
            pbar = ProgressBar(widgets=[Percentage(), Bar()],
                               maxval=len(data.track_names))
            pbar.start()
        except Exception as e:
            pass

        if config.learn_phase:
            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            h5f_path = os.path.join(self.analysisPath,
                                    "distribution_ir_%s.hdf5" % name)
            h5file = h5py.File(h5f_path, "w")
            h5real = h5file.create_group("real")
            h5imag = h5file.create_group("imag")

            plt.figure(figsize=(15, 15))
            plt.suptitle(name)
            ax1 = plt.subplot(231)
            bins = self._do_distribution_plot(pbar, h5real, data, spectrograms,
                                              None, "upper", "real")

            plt.subplot(232, sharey=ax1, sharex=ax1)
            self._do_distribution_plot(pbar, h5real, data, spectrograms,
                                       bins, "center", "real")

            plt.subplot(233, sharey=ax1, sharex=ax1)
            self._do_distribution_plot(pbar, h5real, data, spectrograms,
                                       bins, "lower", "real")

            ax1 = plt.subplot(234)
            bins = self._do_distribution_plot(pbar, h5imag, data, spectrograms,
                                              None, "upper", "imag")

            plt.subplot(235, sharey=ax1, sharex=ax1)
            self._do_distribution_plot(pbar, h5imag, data, spectrograms,
                                       bins, "center", "imag")

            plt.subplot(236, sharey=ax1, sharex=ax1)
            self._do_distribution_plot(pbar, h5imag, data, spectrograms,
                                       bins, "lower", "imag")

            h5file.close()

            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            plt.savefig(os.path.join(self.analysisPath,
                                     "distribution_%s_ir.png" % name))
            plt.close()
        else:
            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            h5f_path = os.path.join(self.analysisPath,
                                    "distribution_amplitude_%s.hdf5" % name)
            h5file = h5py.File(h5f_path, "w")

            plt.figure(figsize=(15, 15))
            plt.suptitle(name)
            ax1 = plt.subplot(131)
            bins = self._do_distribution_plot(pbar, h5file, data, spectrograms,
                                              None, "upper")

            plt.subplot(132, sharey=ax1, sharex=ax1)
            self._do_distribution_plot(pbar, h5file, data, spectrograms,
                                       bins, "center")

            plt.subplot(133, sharey=ax1, sharex=ax1)
            self._do_distribution_plot(pbar, h5file, data, spectrograms,
                                       bins, "lower")
            h5file.close()

            if not os.path.exists(self.analysisPath):
                os.mkdir(self.analysisPath)
            plt.savefig(os.path.join(self.analysisPath,
                                     "distribution_%s_amplitude.png" % name))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse", "-a", default=None, type=str,
                        help="analysis to be executed")
    parser.add_argument("--save", "-s", action='store_true',
                        help="save analysis output to file")
    parser.add_argument("args", nargs="*", default=[])

    arguments = parser.parse_args()

    analysis = Analysis()
    analysis.run(arguments.analyse, arguments.save, arguments.args)
