#!/usr/bin/python3
"""
Utilities to convert wav files to spectrograms and back

"""

import librosa
import numpy as np
import warnings
import skimage.io as io
from os.path import basename
from math import ceil
import argparse
from matplotlib.cm import get_cmap
import console


def load_audio_file(file_path):
    audio, sample_rate = librosa.load(file_path)
    return audio, sample_rate


def save_audio_file(audio_file, file_path, sample_rate):
    librosa.output.write_wav(file_path, audio_file, sample_rate, norm=False)
    console.info("Wrote audio file to", file_path)


def expand_to_grid(spectrogram, grid_size, channels):
    # crop along both axes
    new_y = ceil(spectrogram.shape[1] / grid_size) * grid_size
    new_x = ceil(spectrogram.shape[0] / grid_size) * grid_size
    new_spectrogram = np.zeros((new_x, new_y, channels))
    new_spectrogram[:spectrogram.shape[0],
                    :spectrogram.shape[1], :] = spectrogram
    return new_spectrogram


def stft_to_amplitude(stft):
    spectrogram = np.log1p(np.abs(stft))
    return np.array(spectrogram)[:, :, np.newaxis]


def stft_to_real_and_imag(stft):
    real = np.real(stft)
    imag = np.imag(stft)
    spectrogram = np.zeros((stft.shape[0], stft.shape[1], 2))
    spectrogram[:, :, 0] = real
    spectrogram[:, :, 1] = imag
    return spectrogram


def real_and_imag_to_stft(spectrogram):
    real = spectrogram[:, :, 0]
    imag = spectrogram[:, :, 1]
    return real + imag * 1j


def audio_file_to_stft(audio_file, fft_window_size):
    return librosa.stft(audio_file, fft_window_size)


def audio_file_to_spectrogram(audio_file, fft_window_size, learn_phase=False):
    spectrogram = audio_file_to_stft(audio_file, fft_window_size)
    if learn_phase:
        return stft_to_real_and_imag(spectrogram)
    else:
        return stft_to_amplitude(spectrogram)


def spectrogram_to_audio_file(spectrogram, fft_window_size,
                              phase_iterations=10, learn_phase=False):
    if learn_phase:
        stft_matrix = real_and_imag_to_stft(spectrogram)
        audio = librosa.istft(stft_matrix)
    else:
        # phase reconstruction with successive approximation
        # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410  # noqa: E501
        # for the algorithm used
        spectrogram = spectrogram[:, :, 0]

        amplitude = np.exp(spectrogram) - 1
        for i in range(phase_iterations):
            if i == 0:
                a = np.random.random_sample(amplitude.shape) + 1j
                b = 2 * np.pi * np.random.random_sample(amplitude.shape)
                reconstruction = a * b - np.pi
            else:
                reconstruction = librosa.stft(audio, fft_window_size)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio


def load_spectrogram(file_path):
    filename = basename(file_path)
    if file_path.index("sample_rate") < 0:
        console.warn("Sample rate should be specified in file name", file_path)
        sample_rate = 22050
    else:
        sample_rate = int(filename[filename.index(
            "sample_rate=") + 11:filename.index(").png")])
    console.info("Using sample rate : " + str(sample_rate))
    image = io.imread(file_path, as_grey=True)
    return image / np.max(image), sample_rate


def save_spectrogram(spectrogram, file_path, learn_phase=False):
    if learn_phase:
        spectrogram = stft_to_amplitude(
            real_and_imag_to_stft(spectrogram))

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
        io.imsave(file_path, image)
    console.log("Saved image to", file_path)


def file_suffix(title, **kwargs):
    args = "".join(sorted([", " + i + "=" + str(kwargs[i]) for i in kwargs]))
    return " ({}{})".format(title, args)


def handle_audio(file_path, args):
    console.h1("Creating Spectrogram")
    input_filename = basename(file_path)

    console.info("Attempting to read from " + file_path)
    audio, sample_rate = load_audio_file(file_path)
    console.info("Max of audio file is " + str(np.max(audio)))
    spectrogram = audio_file_to_spectrogram(audio, fft_window_size=args.fft)
    spectrogram_filename = input_filename + \
        file_suffix("Input Spectrogram", fft=args.fft,
                    iter=args.iter, sample_rate=sample_rate) + ".png"

    save_spectrogram(spectrogram, spectrogram_filename)

    print()
    console.wait("Saved Spectrogram; press Enter to continue...")
    print()

    handle_image(spectrogram_filename, args)


def handle_image(filename, args):
    console.h1("Reconstructing Audio from Spectrogram")

    spectrogram, sample_rate = load_spectrogram(filename)
    audio = spectrogram_to_audio_file(
        spectrogram, fft_window_size=args.fft, phase_iterations=args.iter)

    sanity_check = audio_file_to_spectrogram(audio, fft_window_size=args.fft)
    outname = filename \
        + file_suffix("Output Spectrogram",
                      fft=args.fft,
                      iter=args.iter,
                      sample_rate=sample_rate) \
        + ".png"
    save_spectrogram(sanity_check, outname)

    outname = filename \
        + file_suffix("Output", fft=args.fft, iter=args.iter) \
        + ".wav", sample_rate
    save_audio_file(audio, outname)


if __name__ == "__main__":
    # Test code for experimenting with modifying vocals in image processors
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
            handle_audio(f, args)
        elif (f.endswith(".png")):
            handle_image(f, args)
