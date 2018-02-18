import conversion
import argparse
import os
import h5py
import numpy as np

BATCH_NORMALIZATION_INDEX = "batch_normalization_{}"
CONV2D_INDEX = "conv2d_{}"

BATCH_LAYERS = 4
CONV2D_LAYERS = 12


def analyse_spectrogram(filePath):
    audio, sampleRate = conversion.loadAudioFile(filePath)
    spectrogram, phase = conversion.audioFileToSpectrogram(audio, 1536)
    conversion.saveSpectrogram(spectrogram,
                               filePath.replace(".wav", "_spectrogram.png"))


def analyse_weights(directory):
    weights = read_weights_from_dir(directory)

    for i in range(0, len(weights)-1):
        print("\nComparing weights of epoch %d with epoch %d\n" % (i+1, i+2))
        beta_dev, gamma_dev, moving_mean_dev, moving_var_dev \
            = compare_batch_normalization(weights[i], weights[i+1])
        bias_dev, kernel_dev = compare_conv2d(weights[i], weights[i+1])

        print("Epochs %d to %d: Batch normalization mean deviation: "
              "beta=%f, gamma=%f, moving mean=%f moving variance=%f"
              % (i+1, i+2, beta_dev, gamma_dev,
                 moving_mean_dev, moving_var_dev))
        print("Epochs %d to %d: Conv2d mean deviation: bias=%f, kernel=%f"
              % (i+1, i+2, bias_dev, kernel_dev))


def compare_batch_normalization(weight1, weight2):
    mean_dev_beta = []
    mean_dev_gamma = []
    mean_dev_mov_mean = []
    mean_dev_mov_var = []
    for i in range(1, BATCH_LAYERS+1):
        (beta1, gamma1, moving_mean1, moving_variance1) \
            = get_batch_normalization_data(1, weight1)
        (beta2, gamma2, moving_mean2, moving_variance2) \
            = get_batch_normalization_data(1, weight2)
        beta_diff = np.sum(abs(np.subtract(beta1, beta2)))/beta1.shape[0]
        gamma_diff = np.sum(abs(np.subtract(gamma1, gamma2)))/gamma1.shape[0]
        moving_mean_diff \
            = np.sum(abs(np.subtract(moving_mean1, moving_mean2))) / \
            moving_mean1.shape[0]
        moving_variance_diff \
            = np.sum(abs(np.subtract(moving_variance1, moving_variance2))) / \
            moving_variance1.shape[0]
        mean_dev_beta.append(beta_diff)
        mean_dev_gamma.append(gamma_diff)
        mean_dev_mov_mean.append(moving_mean_diff)
        mean_dev_mov_var.append(moving_variance_diff)
        print("  Mean deviation for batch normalization layer %d: "
              "beta=%f, gamma=%f, moving mean=%f, moving variance=%f"
              % (i, beta_diff, gamma_diff,
                 moving_mean_diff, moving_variance_diff))

    beta_dev = np.sum(mean_dev_beta)/len(mean_dev_beta)
    gamma_dev = np.sum(mean_dev_gamma)/len(mean_dev_gamma)
    moving_mean_dev = np.sum(mean_dev_mov_mean)/len(mean_dev_mov_mean)
    moving_var_dev = np.sum(mean_dev_mov_var)/len(mean_dev_mov_var)

    return beta_dev, gamma_dev, moving_mean_dev, moving_var_dev


def compare_conv2d(weight1, weight2):
    mean_dev_bias = []
    mean_dev_kernel = []
    for i in range(1, CONV2D_LAYERS+1):
        (bias1, kernel1) = get_conv2d_layer_data(1, weight1)
        (bias2, kernel2) = get_conv2d_layer_data(1, weight2)
        bias_diff = np.sum(abs(np.subtract(bias1, bias2)))/bias1.shape[0]
        kernel_diff = np.sum(abs(np.subtract(kernel1, kernel2))) / \
            np.prod(kernel1.shape)
        mean_dev_bias.append(bias_diff)
        mean_dev_kernel.append(kernel_diff)
        print("  Mean deviation for conv2D layer %d: bias=%f, kernel=%f"
              % (i, bias_diff, kernel_diff))

    bias_dev = np.sum(mean_dev_bias)/len(mean_dev_bias)
    kernel_dev = np.sum(mean_dev_kernel)/len(mean_dev_kernel)

    return bias_dev, kernel_dev


def get_batch_normalization_data(number, weights):
    index = BATCH_NORMALIZATION_INDEX.format(number)
    tmp = weights[index][index]["beta:0"]
    beta = np.zeros(tmp.shape)
    tmp.read_direct(beta)
    tmp = weights[index][index]["gamma:0"]
    gamma = np.zeros(tmp.shape)
    tmp.read_direct(gamma)
    tmp = weights[index][index]["moving_mean:0"]
    moving_mean = np.zeros(tmp.shape)
    tmp.read_direct(moving_mean)
    tmp = weights[index][index]["moving_variance:0"]
    moving_variance = np.zeros(tmp.shape)
    tmp.read_direct(moving_variance)

    return beta, gamma, moving_mean, moving_variance


def get_conv2d_layer_data(number, weights):
    index = CONV2D_INDEX.format(number)
    tmp = weights[index][index]["bias:0"]
    bias = np.zeros(tmp.shape)
    tmp.read_direct(bias)
    tmp = weights[index][index]["kernel:0"]
    kernel = np.zeros(tmp.shape)
    tmp.read_direct(kernel)

    return bias, kernel


def print_h5_structure(weights):

    def print_name(name):
        print(name)

    weights.visit(print_name)


def read_weights_from_dir(directory):
    def checkFilename(f):
        return (f.endswith(".h5") or f.endswith("hdf5")) \
               and not f.startswith(".")

    weights = []

    for dirPath, dirNames, fileNames in os.walk(directory):
        filteredFiles = filter(checkFilename, fileNames)

        for fileName in filteredFiles:
            path = os.path.join(directory, fileName)
            weight = h5py.File(path, "r")
            weights.append(weight)

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str,
                        help="file for spectrogram analysis")
    parser.add_argument("--weights", default=None, type=str,
                        help="directory for weight analysis")
    args = parser.parse_args()
    if args.file:
        analyse_spectrogram(args.file)
    if args.weights:
        analyse_weights(args.weights)
