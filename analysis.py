import argparse
import datetime

import conversion
import os
import numpy as np
from config import Config
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.initializers import Ones, Zeros
import h5py
from chopper import Chopper

BATCH_NORMALIZATIONINDEX = "batch_normalization_{}"
CONV2DINDEX = "conv2d_{}"

BATCH_LAYERS = 4
CONV2D_LAYERS = 12


class Analysis:
    def __init__(self):
        self.config = Config()
        self.analyse = "spectrograms"
        self.save = True
        self.analysisPath = self.config.analysis_path
        self.content = "Analyse " + self.analyse + "\n"

    def get(self):
        return getattr(self, self.analyse)

    def run(self, analyse, save, args):
        self.analyse = analyse
        self.save = save

        config_str = str(self.config)
        print(config_str)

        print(self.content)
        analyse = self.get()
        analyse(*args)

    def slices(self, file):
        spectrogram = self._create_spectrogram_from_file(file)
        currMin = float("inf")
        currMinI = -1
        currMax = 0
        currMaxI = 1

        chop = Chopper.get()

        slices = chop(spectrogram)
        for i in range(0, len(slices)):
            smin = np.min(slices[i])
            smax = np.max(slices[i])
            if smin < currMin:
                currMin = smin
                currMinI = i
            if smax > currMax:
                currMax = smax
                currMaxI = i

        meanDev = np.sum(slices) / (len(slices) * np.prod(slices[0].shape))

        self.write("Minimum at %d with %f" % (currMinI, currMin))
        self.write("Maximum at %d with %f" % (currMaxI, currMax))
        self.write("Mean deviation %f" % meanDev)

        countDevSum = 0
        countDevMax = 0

        for slice in slices:
            if np.sum(slice) / np.prod(slice.shape) > meanDev:
                countDevSum += 1
            if np.max(slice) > meanDev:
                countDevMax += 1

        self.write("Count sum above mean deviation is %d of %d"
                   % (countDevSum, len(slices)))
        self.write("Count max above mean deviation is %d of %d"
                   % (countDevMax, len(slices)))

        self._save_analysis()

    def spectrograms(self, directory):

        data = self._read_spectrograms_from_dir(directory)

        counts = [0, 0, 0]
        desc = ["upper", "center", "lower"]

        for (spectrogram, name) in data:

            means = []
            window = spectrogram.shape[0]//2
            upperWindow = spectrogram[0:window]
            means.append(np.sum(upperWindow) / np.prod(upperWindow.shape))
            centerWindow = spectrogram[window//2: window//2 + window]
            means.append(np.sum(centerWindow) / np.prod(centerWindow.shape))
            lowerWindow = spectrogram[-window:]
            means.append(np.sum(lowerWindow) / np.prod(lowerWindow.shape))

            best = np.argmax(means)
            counts[best] += 1

            self.write("{:50s} best mean {:8f} in {:s} window"
                       .format(name, np.max(means), desc[best]))

        self.write("\nStatistics\n")
        self.write("%d spectrograms analysed" % len(data))
        self.write("%d have the highest mean in %s window"
                   % (counts[0], desc[0]))
        self.write("%d have the highest mean in %s window"
                   % (counts[1], desc[1]))
        self.write("%d have the highest mean in %s window"
                   % (counts[2], desc[2]))
        self._save_analysis()

    def weights(self, directory):
        weights = self._read_weights_from_dir(directory)

        for i in range(0, len(weights) - 1):
            self.write("Comparing weights of epoch %d with epoch %d"
                       % (i + 1, i + 2))

            betaDev, gammaDev, movingMeanDev, movingVarDev \
                = self._compare_batch_normalization(weights[i], weights[i + 1])
            biasDev, kernelDev \
                = self._compare_conv2d(weights[i], weights[i + 1])

            self.write("Epochs %d to %d: Batch normalization mean deviation: "
                       "beta=%f, gamma=%f, moving mean=%f moving variance=%f"
                       % (i + 1, i + 2, betaDev, gammaDev,
                          movingMeanDev, movingVarDev))
            self.write("Epochs %d to %d: Conv2d mean deviation: "
                       "bias=%f, kernel=%f"
                       % (i + 1, i + 2, biasDev, kernelDev))

        self._save_analysis()

    # If the output is close to the input,
    # the naive solution would be to just pass the input through the network.
    # If the output is mostly close to 0,
    # then the naive solution would be to always return 0.
    #
    # Calculate the loss of these two naive solutions.
    # The real loss of the network should be below these values.
    def naive_solutions(self, file):
        h5f = h5py.File(file, "r")
        x = h5f["x"][:]
        y = h5f["y"][:]

        mashup = Input(shape=(None, None, 1), name='input')

        # model with zero output
        conv0 = Conv2D(1, 1, activation='linear',
                       kernel_initializer=Zeros(), padding='same')(mashup)
        model0 = Model(inputs=mashup, outputs=conv0)
        model0.compile(loss='mean_squared_error', optimizer='adam')
        model0.summary(line_length=150)

        # model with output=input
        conv1 = Conv2D(1, 1, activation='linear',
                       kernel_initializer=Ones(), padding='same')(mashup)
        model1 = Model(inputs=mashup, outputs=conv1)
        model1.compile(loss='mean_squared_error', optimizer='adam')
        model1.summary(line_length=150)

        error0 = model0.evaluate(x, y, batch_size=8)
        error1 = model1.evaluate(x, y, batch_size=8)

        self.write("MSE for output=all_zeros: %f" % error0)
        self.write("MSE for output=input: %f" % error1)
        self._save_analysis()

    def _compare_batch_normalization(self, weight1, weight2):
        meanDevBeta = []
        meanDevGamma = []
        meanDevMovMean = []
        meanDevMovVar = []
        for i in range(1, BATCH_LAYERS + 1):
            (beta1, gamma1, movingMean1, movingVariance1) \
                = self._get_batch_normalization_data(1, weight1)
            (beta2, gamma2, movingMean2, movingVariance2) \
                = self._get_batch_normalization_data(1, weight2)
            betaDiff = np.sum(abs(np.subtract(beta1, beta2))) / beta1.shape[0]
            gammaDiff = np.sum(abs(np.subtract(gamma1, gamma2))) \
                / gamma1.shape[0]
            movingMeanDiff \
                = np.sum(abs(np.subtract(movingMean1, movingMean2))) / \
                movingMean1.shape[0]
            movingVarianceDiff \
                = np.sum(abs(np.subtract(movingVariance1, movingVariance2))) \
                / movingVariance1.shape[0]
            meanDevBeta.append(betaDiff)
            meanDevGamma.append(gammaDiff)
            meanDevMovMean.append(movingMeanDiff)
            meanDevMovVar.append(movingVarianceDiff)
            self.write("  Mean deviation for batch normalization layer %d: "
                       "beta=%f, gamma=%f, moving mean=%f, moving variance=%f"
                       % (i, betaDiff, gammaDiff,
                          movingMeanDiff, movingVarianceDiff))

        betaDev = np.sum(meanDevBeta) / len(meanDevBeta)
        gammaDev = np.sum(meanDevGamma) / len(meanDevGamma)
        movingMeanDev = np.sum(meanDevMovMean) / len(meanDevMovMean)
        movingVarDev = np.sum(meanDevMovVar) / len(meanDevMovVar)

        return betaDev, gammaDev, movingMeanDev, movingVarDev

    def _compare_conv2d(self, weight1, weight2):
        meanDevBias = []
        meanDevKernel = []
        for i in range(1, CONV2D_LAYERS + 1):
            (bias1, kernel1) = self._get_conv2d_layer_data(1, weight1)
            (bias2, kernel2) = self._get_conv2d_layer_data(1, weight2)
            biasDiff = np.sum(abs(np.subtract(bias1, bias2))) / bias1.shape[0]
            kernelDiff = np.sum(abs(np.subtract(kernel1, kernel2))) / \
                np.prod(kernel1.shape)
            meanDevBias.append(biasDiff)
            meanDevKernel.append(kernelDiff)
            self.write("  Mean deviation for conv2D layer %d: "
                       "bias=%f, kernel=%f"
                       % (i, biasDiff, kernelDiff))

        biasDev = np.sum(meanDevBias) / len(meanDevBias)
        kernelDev = np.sum(meanDevKernel) / len(meanDevKernel)

        return biasDev, kernelDev

    def _get_batch_normalization_data(self, number, weights):
        index = BATCH_NORMALIZATIONINDEX.format(number)
        tmp = weights[index][index]["beta:0"]
        beta = np.zeros(tmp.shape)
        tmp.read_direct(beta)
        tmp = weights[index][index]["gamma:0"]
        gamma = np.zeros(tmp.shape)
        tmp.read_direct(gamma)
        tmp = weights[index][index]["moving_mean:0"]
        movingMean = np.zeros(tmp.shape)
        tmp.read_direct(movingMean)
        tmp = weights[index][index]["moving_variance:0"]
        movingVariance = np.zeros(tmp.shape)
        tmp.read_direct(movingVariance)

        return beta, gamma, movingMean, movingVariance

    def _get_conv2d_layer_data(self, number, weights):
        index = CONV2DINDEX.format(number)
        tmp = weights[index][index]["bias:0"]
        bias = np.zeros(tmp.shape)
        tmp.read_direct(bias)
        tmp = weights[index][index]["kernel:0"]
        kernel = np.zeros(tmp.shape)
        tmp.read_direct(kernel)

        return bias, kernel

    def _print_h5_structure(self, weights):

        def print_name(name):
            print(name)

        weights.visit(print_name)

    def _read_weights_from_dir(self, directory):
        def check_filename(f):
            return (f.endswith(".h5") or f.endswith("hdf5")) \
                   and not f.startswith(".")

        weights = []

        for dirPath, dirNames, fileNames in os.walk(directory):
            filteredFiles = filter(check_filename, fileNames)

            for fileName in filteredFiles:
                path = os.path.join(directory, fileName)
                weight = h5py.File(path, "r")
                weights.append(weight)

        return weights

    def _read_spectrograms_from_dir(self, directory):
        def check_filename(f):
            return f.endswith(".wav") and not f.startswith(".")

        data = []

        for dirPath, dirNames, fileNames in os.walk(directory):
            filteredFiles = filter(check_filename, fileNames)

            for fileName in filteredFiles:
                path = os.path.join(directory, fileName)
                self.write("creating spectrogram for %s" % fileName, True)
                spectrogram = self._create_spectrogram_from_file(path)
                self._spectrogram_info(spectrogram)
                data.append((spectrogram, fileName))

        return data

    def _create_spectrogram_from_file(self, filePath):
        audio, sampleRate = conversion.loadAudioFile(filePath)
        spectrogram, phase = conversion.audioFileToSpectrogram(audio, 1536)

        return spectrogram

    def write(self, message, printAnyway=False):
        if self.save:
            self.content += "\n" + message
            if printAnyway:
                print(message)
        else:
            print(message)

    def _save_analysis(self):
        print("\nAnalysis complete")
        if self.save:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            if not os.path.exists(self.analysisPath):
                os.makedirs(self.analysisPath)

            path = self.analysisPath + "/" + self.analyse + date + ".txt"
            with open(path, "w") as f:
                f.write(self.content)

    def _spectrogram_info(self, spectrogram):
        spectrum = spectrogram
        self.write("Range of spectrogram is " +
                   str(np.min(spectrum)) + " -> " + str(np.max(spectrum)))
        image = np.clip((spectrum - np.min(spectrum)) /
                        (np.max(spectrum) - np.min(spectrum)), 0, 1)
        self.write("Shape of spectrogram is (%d, %d)"
                   % (image.shape[0], image.shape[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse", default=None, type=str,
                        help="analysis to be executed")
    parser.add_argument("--save", action='store_true',
                        help="save analysis output to file")
    parser.add_argument("args", nargs="*", default=[])

    arguments = parser.parse_args()

    analysis = Analysis()
    analysis.run(arguments.analyse, arguments.save, arguments.args)
