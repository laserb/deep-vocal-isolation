import librosa
import numpy as np
import warnings
import skimage.io as io
from os.path import basename
from math import ceil
import argparse
from matplotlib.cm import get_cmap
import console


def loadAudioFile(filePath):
    audio, sampleRate = librosa.load(filePath)
    return audio, sampleRate


def saveAudioFile(audioFile, filePath, sampleRate):
    librosa.output.write_wav(filePath, audioFile, sampleRate, norm=False)
    console.info("Wrote audio file to", filePath)


def expandToGrid(spectrogram, gridSize, channels):
    # crop along both axes
    newY = ceil(spectrogram.shape[1] / gridSize) * gridSize
    newX = ceil(spectrogram.shape[0] / gridSize) * gridSize
    newSpectrogram = np.zeros((newX, newY, channels))
    newSpectrogram[:spectrogram.shape[0],
                   :spectrogram.shape[1], :] = spectrogram
    return newSpectrogram


def stftToAmplitude(stft):
    spectrogram = np.log1p(np.abs(stft))
    return np.array(spectrogram)[:, :, np.newaxis]


def stftToRealAndImag(stft):
    real = np.real(stft)
    imag = np.imag(stft)
    spectrogram = np.zeros((stft.shape[0], stft.shape[1], 2))
    spectrogram[:, :, 0] = real
    spectrogram[:, :, 1] = imag
    return spectrogram


def realAndImagToStft(spectrogram):
    real = spectrogram[:, :, 0]
    imag = spectrogram[:, :, 1]
    return real + imag * 1j


def audioFileToStft(audioFile, fftWindowSize):
    return librosa.stft(audioFile, fftWindowSize)


def audioFileToSpectrogram(audioFile, fftWindowSize, learn_phase=False):
    spectrogram = audioFileToStft(audioFile, fftWindowSize)
    if learn_phase:
        return stftToRealAndImag(spectrogram)
    else:
        return stftToAmplitude(spectrogram)


def spectrogramToAudioFile(spectrogram, fftWindowSize,
                           phaseIterations=10, learnPhase=False):
    if learnPhase:
        stftMatrix = realAndImagToStft(spectrogram)
        audio = librosa.istft(stftMatrix)
    else:
        # phase reconstruction with successive approximation
        # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410  # noqa: E501
        # for the algorithm used
        spectrogram = spectrogram[:, :, 0]

        amplitude = np.exp(spectrogram) - 1
        for i in range(phaseIterations):
            if i == 0:
                a = np.random.random_sample(amplitude.shape) + 1j
                b = 2 * np.pi * np.random.random_sample(amplitude.shape)
                reconstruction = a * b - np.pi
            else:
                reconstruction = librosa.stft(audio, fftWindowSize)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio


def loadSpectrogram(filePath):
    fileName = basename(filePath)
    if filePath.index("sampleRate") < 0:
        console.warn("Sample rate should be specified in file name", filePath)
        sampleRate = 22050
    else:
        sampleRate = int(fileName[fileName.index(
            "sampleRate=") + 11:fileName.index(").png")])
    console.info("Using sample rate : " + str(sampleRate))
    image = io.imread(filePath, as_grey=True)
    return image / np.max(image), sampleRate


def saveSpectrogram(spectrogram, filePath, learnPhase=False):
    if learnPhase:
        spectrogram = stftToAmplitude(
            realAndImagToStft(spectrogram))

    spectrogram = spectrogram[:, :, 0]

    cm_hot = get_cmap('magma')
    spectrum = spectrogram
    console.info("Range of spectrum is " +
                 str(np.min(spectrum)) + " -> " + str(np.max(spectrum)))
    image = np.clip((spectrum - np.min(spectrum)) /
                    (np.max(spectrum) - np.min(spectrum)), 0, 1)
    console.info("Shape of spectrum is", image.shape)
    # Low-contrast image warnings are not helpful, tyvm
    with warnings.catch_warnings():
        image = cm_hot(image)
        warnings.simplefilter("ignore")
        io.imsave(filePath, image)
    console.log("Saved image to", filePath)


def fileSuffix(title, **kwargs):
    args = "".join(sorted([", " + i + "=" + str(kwargs[i]) for i in kwargs]))
    return " ({}{})".format(title, args)


def handleAudio(filePath, args):
    console.h1("Creating Spectrogram")
    INPUT_FILE = filePath
    INPUT_FILENAME = basename(INPUT_FILE)

    console.info("Attempting to read from " + INPUT_FILE)
    audio, sampleRate = loadAudioFile(INPUT_FILE)
    console.info("Max of audio file is " + str(np.max(audio)))
    spectrogram = audioFileToSpectrogram(audio, fftWindowSize=args.fft)
    SPECTROGRAM_FILENAME = INPUT_FILENAME + \
        fileSuffix("Input Spectrogram", fft=args.fft,
                   iter=args.iter, sampleRate=sampleRate) + ".png"

    saveSpectrogram(spectrogram, SPECTROGRAM_FILENAME)

    print()
    console.wait("Saved Spectrogram; press Enter to continue...")
    print()

    handleImage(SPECTROGRAM_FILENAME, args)


def handleImage(fileName, args):
    console.h1("Reconstructing Audio from Spectrogram")

    spectrogram, sampleRate = loadSpectrogram(fileName)
    audio = spectrogramToAudioFile(
        spectrogram, fftWindowSize=args.fft, phaseIterations=args.iter)

    sanityCheck = audioFileToSpectrogram(audio, fftWindowSize=args.fft)
    outname = fileName \
        + fileSuffix("Output Spectrogram",
                     fft=args.fft,
                     iter=args.iter,
                     sampleRate=sampleRate) \
        + ".png"
    saveSpectrogram(sanityCheck, outname)

    outname = fileName \
        + fileSuffix("Output", fft=args.fft, iter=args.iter) \
        + ".wav", sampleRate
    saveAudioFile(audio, outname)


if __name__ == "__main__":
    # Test code for experimenting with modifying acapellas in image processors
    # (and generally testing the reconstruction pipeline)
    parser = argparse.ArgumentParser(
        description="Convert image files to audio and audio files to images")
    parser.add_argument("--fft", default=1536, type=int,
                        help="Size of FFT windows")
    parser.add_argument("--iter", default=10, type=int,
                        help="Number of iterations to use "
                             "for phase reconstruction")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    for f in args.files:
        if (f.endswith(".mp3") or f.endswith(".wav")):
            handleAudio(f, args)
        elif (f.endswith(".png")):
            handleImage(f, args)
