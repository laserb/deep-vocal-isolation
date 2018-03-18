import argparse
import datetime

import conversion
import os
import numpy as np
from config import config
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.initializers import Ones, Zeros
import h5py
from chopper import Chopper
from normalizer import Normalizer
from acapellabot import AcapellaBot
from checkpointer import ErrorVisualization
from loss import Loss
from data import Data

BATCH_NORMALIZATIONINDEX = "batch_normalization_{}"
CONV2DINDEX = "conv2d_{}"

BATCH_LAYERS = 4
CONV2D_LAYERS = 12


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

    def slices(self, file):
        spectrogram = self._create_spectrogram_from_file(file)

        currMin = [float("inf"), float("inf")]
        currMinI = [-1, -1]
        currMax = [0, 0]
        currMaxI = [1, 1]
        meanDev = [0, 0]
        countDevSum = [0, 0]
        countDevMax = [0, 0]

        chop = Chopper().get(False)

        slices = chop(spectrogram)
        for i in range(0, len(slices)):

            if self.config.learn_phase:
                currMinI[0], currMin[0] = \
                    self._get_current_min(slices[i][:, :, 0],
                                          currMinI[0], currMin[0], i)
                currMaxI[0], currMax[0] = \
                    self._get_current_max(slices[i][:, :, 0],
                                          currMaxI[0], currMax[0], i)
                currMinI[1], currMin[1] = \
                    self._get_current_min(slices[i][:, :, 1],
                                          currMinI[1], currMin[1], i)
                currMaxI[1], currMax[1] = \
                    self._get_current_max(slices[i][:, :, 1],
                                          currMaxI[1], currMax[1], i)
            else:
                currMinI[0], currMin[0] = \
                    self._get_current_min(slices[i],
                                          currMinI[0], currMin[0], i)
                currMaxI[0], currMax[0] = \
                    self._get_current_max(slices[i],
                                          currMaxI[0], currMax[0], i)

        if self.config.learn_phase:
            s = np.array(slices)[:]
            meanDev[0] = np.sum(s[:, :, :, 0]) / \
                (len(slices) * np.prod(slices[0][:, :, 0].shape))
            meanDev[1] = np.sum(s[:, :, :, 1]) / \
                (len(slices) * np.prod(slices[0][:, :, 0].shape))
        else:
            meanDev[0] = np.sum(slices) / (len(slices) *
                                           np.prod(slices[0].shape))

        for slice in slices:
            if self.config.learn_phase:
                countDevSum[0] += self._get_count_dev_sum(slice[:, :, 0],
                                                          meanDev[0])
                countDevMax[0] += self._get_count_dev_max(slice[:, :, 0],
                                                          meanDev[0])
                countDevSum[1] += self._get_count_dev_sum(slice[:, :, 1],
                                                          meanDev[1])
                countDevMax[1] += self._get_count_dev_max(slice[:, :, 1],
                                                          meanDev[1])
            else:
                countDevSum[0] += self._get_count_dev_sum(slice, meanDev[0])
                countDevMax[0] += self._get_count_dev_max(slice, meanDev[0])

        self._write_slices_statistics(currMinI[0], currMin[0], currMaxI[0],
                                      currMax[0], meanDev[0],
                                      countDevSum[0], countDevMax[0],
                                      len(slices))
        if self.config.learn_phase:
            self._write("")
            self._write_slices_statistics(currMinI[1], currMin[1], currMaxI[1],
                                          currMax[1], meanDev[1],
                                          countDevSum[1], countDevMax[1],
                                          len(slices))

        self._save_analysis()

    def _get_count_dev_sum(self, slice, meanDev):
        if np.sum(slice) / np.prod(slice.shape) > meanDev:
            return 1
        else:
            return 0

    def _get_count_dev_max(self, slice, meanDev):
        if np.max(slice) > meanDev:
            return 1
        else:
            return 0

    def _get_current_min(self, spectrogram, currMinI, currMin, index):
        smin = np.min(spectrogram)

        if smin < currMin:
            return index, smin
        else:
            return currMinI, currMin

    def _get_current_max(self, spectrogram, currMaxI, currMax, index):
        smax = np.max(spectrogram)

        if smax > currMax:
            return index, smax
        else:
            return currMaxI, currMax

    def _write_slices_statistics(self, currMinI, currMin,
                                 currMaxI, currMax, meanDev,
                                 countDevSum, countDevMax, slicesLength):
        self._write("Minimum at %d with %f" % (currMinI, currMin))
        self._write("Maximum at %d with %f" % (currMaxI, currMax))
        self._write("Mean deviation %f" % meanDev)

        self._write("Count sum above mean deviation is %d of %d"
                    % (countDevSum, slicesLength))
        self._write("Count max above mean deviation is %d of %d"
                    % (countDevMax, slicesLength))

    def spectrograms(self, directory):

        data = self._read_spectrograms_from_dir(directory)

        counts = [[0, 0, 0], [0, 0, 0]]
        desc = ["upper", "center", "lower"]

        for (spectrogram, name) in data:

            if self.config.learn_phase:
                meansReal = self._calculate_spectrogram_windows(
                    spectrogram[:, :, 0])
                meansImag = self._calculate_spectrogram_windows(
                    spectrogram[:, :, 1])

                bestReal = np.argmax(meansReal)
                bestImag = np.argmax(meansImag)

                counts[0][bestReal] += 1
                counts[1][bestImag] += 1

                self._write("{:50s} best real mean {:8f} in {:s} window"
                            .format(name, np.max(meansReal), desc[bestReal]))
                self._write("{:50s} best imag mean {:8f} in {:s} window"
                            .format(name, np.max(meansImag), desc[bestImag]))
            else:
                means = self._calculate_spectrogram_windows(spectrogram)

                best = np.argmax(means)
                counts[0][best] += 1

                self._write("{:50s} best mean {:8f} in {:s} window"
                            .format(name, np.max(means), desc[best]))

        self._write("\nStatistics\n")
        self._write_spectrogram_statistics(counts[0], desc, len(data))
        if self.config.learn_phase:
            self._write("")
            self._write_spectrogram_statistics(counts[1], desc, len(data))

        self._save_analysis()

    def _calculate_spectrogram_windows(self, spectrogram):
        means = []
        window = spectrogram.shape[0] // 2
        upperWindow = spectrogram[0:window]
        means.append(np.sum(upperWindow) / np.prod(upperWindow.shape))
        centerWindow = spectrogram[window // 2: window // 2 + window]
        means.append(np.sum(centerWindow) / np.prod(centerWindow.shape))
        lowerWindow = spectrogram[-window:]
        means.append(np.sum(lowerWindow) / np.prod(lowerWindow.shape))

        return means

    def _write_spectrogram_statistics(self, counts, desc, dataLength):
        self._write("%d spectrograms analysed" % dataLength)
        self._write("%d have the highest mean in %s window"
                    % (counts[0], desc[0]))
        self._write("%d have the highest mean in %s window"
                    % (counts[1], desc[1]))
        self._write("%d have the highest mean in %s window"
                    % (counts[2], desc[2]))

    def weights(self, directory):
        weights = self._read_weights_from_dir(directory)

        for i in range(0, len(weights) - 1):
            self._write("Comparing weights of epoch %d with epoch %d"
                        % (i + 1, i + 2))

            betaDev, gammaDev, movingMeanDev, movingVarDev \
                = self._compare_batch_normalization(weights[i], weights[i + 1])
            biasDev, kernelDev \
                = self._compare_conv2d(weights[i], weights[i + 1])

            self._write("Epochs %d to %d: Batch normalization mean deviation: "
                        "beta=%f, gamma=%f, moving mean=%f moving variance=%f"
                        % (i + 1, i + 2, betaDev, gammaDev,
                           movingMeanDev, movingVarDev))
            self._write("Epochs %d to %d: Conv2d mean deviation: "
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
    def naive_solutions(self):
        data = Data()
        # use all data as validation data,
        # they have the right form to analyse
        data.trainingSplit = 0
        mashup, output = data.valid()

        channels = self.config.get_channels()

        input_layer = Input(shape=(None, None, channels), name='input')

        loss = Loss().get()

        # model with zero output
        conv0 = Conv2D(channels, 1, activation='linear',
                       kernel_initializer=Zeros(), padding='same')(input_layer)
        model0 = Model(inputs=input_layer, outputs=conv0)
        model0.compile(loss=loss, optimizer='adam')
        model0.summary(line_length=150)

        # model with output=input
        conv1 = Conv2D(channels, 1, activation='linear',
                       kernel_initializer=Ones(), padding='same')(input_layer)
        model1 = Model(inputs=input_layer, outputs=conv1)
        model1.compile(loss=loss, optimizer='adam')
        model1.summary(line_length=150)

        error0 = model0.evaluate(mashup, output, batch_size=8)
        error1 = model1.evaluate(mashup, output, batch_size=8)

        self._write("MSE for output=all_zeros: %f" % error0)
        self._write("MSE for output=input: %f" % error1)
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
            self._write("  Mean deviation for batch normalization layer %d: "
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
            self._write("  Mean deviation for conv2D layer %d: "
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
                self._write("creating spectrogram for %s" % fileName, True)
                spectrogram = self._create_spectrogram_from_file(path)
                self._spectrogram_info(spectrogram)
                data.append((spectrogram, fileName))

        return data

    def _create_spectrogram_from_file(self, filePath):
        audio, sampleRate = conversion.loadAudioFile(filePath)
        spectrogram = \
            conversion.audioFileToSpectrogram(audio, 1536,
                                              self.config.learn_phase)

        return spectrogram

    def error_images(self):
        acapellabot = AcapellaBot(self.config)
        acapellabot.loadWeights(self.config.weights)

        data = Data()
        xValid, yValid = data.valid()
        acapellabot.xValid, acapellabot.yValid = xValid, yValid

        error_visualization = ErrorVisualization(acapellabot)
        error_visualization.on_epoch_end(-1)

    def _write(self, message, printAnyway=False):
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
        if self.config.learn_phase:
            self._write("Range of the real part is " +
                        str(np.min(spectrogram[:, :, 0])) + " -> " +
                        str(np.max(spectrogram[:, :, 0])))
            self._write("Range of the imag part is " +
                        str(np.min(spectrogram[:, :, 1])) + " -> " +
                        str(np.max(spectrogram[:, :, 1])))
        else:
            self._write("Range of spectrogram is " +
                        str(np.min(spectrogram)) + " -> " +
                        str(np.max(spectrogram)))
        image = np.clip((spectrogram - np.min(spectrogram)) /
                        (np.max(spectrogram) - np.min(spectrogram)), 0, 1)
        self._write("Shape of spectrogram is (%d, %d)"
                    % (image.shape[0], image.shape[1]))

    def _chopper_info(self, mashupSlices, acapellaSlices):
        self._write("chop name: " + self.config.chopname, True)
        self._write("slices created: %d" % len(mashupSlices), True)
        self._write("Shape of first slice: %s"
                    % (mashupSlices[0].shape,), True)
        self._write("Shape of last slice: %s"
                    % (mashupSlices[-1].shape,), True)
        self._write("----------", True)

    def chopper(self, file, chopparams=None):
        spectrogram = self._create_spectrogram_from_file(file)
        self._spectrogram_info(spectrogram)

        chopNames = Chopper().get_all_chop_names()

        if chopparams is not None:
            if isinstance(eval(chopparams), dict):
                params = chopparams
            else:
                params = self.config.chopparams
        else:
            params = self.config.chopparams

        params = eval(params)

        params['upper'] = False
        self.config.chopparams = str(params)
        self._write("\nchop params: " + self.config.chopparams + "\n", True)

        for name in chopNames:
            self.config.chopname = name
            chop = Chopper().get()
            mashupSlices, acapellaSlices = chop(spectrogram, spectrogram)
            self._chopper_info(mashupSlices, acapellaSlices)

        params['upper'] = True
        self.config.chopparams = str(params)
        self._write("\nchop params: " + self.config.chopparams + "\n", True)

        for name in chopNames:
            self.config.chopname = name
            chop = Chopper().get()
            mashupSlices, acapellaSlices = chop(spectrogram, spectrogram)
            self._chopper_info(mashupSlices, acapellaSlices)

        self._save_analysis()

    def normalizer(self, file, learnPhase=False):
        self.config.learn_phase = eval(learnPhase)
        spectrogram = self._create_spectrogram_from_file(file)
        self._spectrogram_info(spectrogram)

        self.config.normalizer = "percentile"
        self.config.normalizer_params = "{'percentile': 95}"
        normalizer = Normalizer()
        normalize = normalizer.get(both=False)
        denormalize = normalizer.get_reverse()

        minS = [0, 0]
        maxS = [0, 0]
        meanS = [0, 0]
        perc = [0, 0]

        percentile = eval(self.config.normalizer_params)['percentile']
        if self.config.learn_phase:
            minS[0] = np.min(spectrogram[:, :, 0])
            maxS[0] = np.max(spectrogram[:, :, 0])
            meanS[0] = np.mean(spectrogram[:, :, 0])
            perc[0] = np.percentile(spectrogram[:, :, 0], percentile)

            minS[1] = np.min(spectrogram[:, :, 1])
            maxS[1] = np.max(spectrogram[:, :, 1])
            meanS[1] = np.mean(spectrogram[:, :, 1])
            perc[1] = np.percentile(spectrogram[:, :, 1], percentile)

            self._write("Original Real: Minimum %f, "
                        "Maximum %f, Mean %f percentile %d value %f"
                        % (minS[0], maxS[0], meanS[0],
                           percentile, perc[0]), printAnyway=True)
            self._write("Original Imag: Minimum %f, "
                        "Maximum %f, Mean %f percentile %d value %f"
                        % (minS[1], maxS[1], meanS[1],
                           percentile, perc[1]), printAnyway=True)

            spectrogram, norm = normalize(spectrogram)

            minS[0] = np.min(spectrogram[:, :, 0])
            maxS[0] = np.max(spectrogram[:, :, 0])
            meanS[0] = np.mean(spectrogram[:, :, 0])
            perc[0] = np.percentile(spectrogram[:, :, 0], percentile)

            minS[1] = np.min(spectrogram[:, :, 1])
            maxS[1] = np.max(spectrogram[:, :, 1])
            meanS[1] = np.mean(spectrogram[:, :, 1])
            perc[1] = np.percentile(spectrogram[:, :, 1], percentile)

            self._write("Normalized Real: Minimum %f, "
                        "Maximum %f, Mean %f percentile %d value %f"
                        % (minS[0], maxS[0], meanS[0],
                           percentile, perc[0]), printAnyway=True)
            self._write("Normalized Imag: Minimum %f, "
                        "Maximum %f, Mean %f percentile %d value %f"
                        % (minS[1], maxS[1], meanS[1],
                           percentile, perc[1]), printAnyway=True)

            spectrogram = denormalize(spectrogram, norm)

            minS[0] = np.min(spectrogram[:, :, 0])
            maxS[0] = np.max(spectrogram[:, :, 0])
            meanS[0] = np.mean(spectrogram[:, :, 0])
            perc[0] = np.percentile(spectrogram[:, :, 0], percentile)

            minS[1] = np.min(spectrogram[:, :, 1])
            maxS[1] = np.max(spectrogram[:, :, 1])
            perc[1] = np.percentile(spectrogram[:, :, 1], percentile)

            self._write("Denormalized Real: Minimum %f, "
                        "Maximum %f, Mean %f percentile %d value %f"
                        % (minS[0], maxS[0], meanS[0],
                           percentile, perc[0]), printAnyway=True)
            self._write("Denormalized Imag: Minimum %f, "
                        "Maximum %f, Mean %f percentile %d value %f"
                        % (minS[1], maxS[1], meanS[1],
                           percentile, perc[1]), printAnyway=True)
        else:
            minS[0] = np.min(spectrogram)
            maxS[0] = np.max(spectrogram)
            meanS[0] = np.mean(spectrogram)
            perc[0] = np.percentile(spectrogram[:, :, 0], percentile)

            self._write("Original: Minimum %f, Maximum %f, Mean %f "
                        "percentile %d value %f"
                        % (minS[0], maxS[0], meanS[0],
                           percentile, perc[0]), printAnyway=True)

            spectrogram, norm = normalize(spectrogram)

            minS[0] = np.min(spectrogram)
            maxS[0] = np.max(spectrogram)
            meanS[0] = np.mean(spectrogram)
            perc[0] = np.percentile(spectrogram[:, :, 0], percentile)

            self._write("Normalized: Minimum %f, Maximum %f, Mean %f "
                        "percentile %d value %f"
                        % (minS[0], maxS[0], meanS[0],
                           percentile, perc[0]), printAnyway=True)

            spectrogram = denormalize(spectrogram, norm)

            minS[0] = np.min(spectrogram)
            maxS[0] = np.max(spectrogram)
            meanS[0] = np.mean(spectrogram)
            perc[0] = np.percentile(spectrogram[:, :, 0], percentile)

            self._write("Denormalized Minimum %f, Maximum %f, Mean %f "
                        "percentile %d value %f"
                        % (minS[0], maxS[0], meanS[0],
                           percentile, perc[0]), printAnyway=True)

        self._save_analysis()


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
