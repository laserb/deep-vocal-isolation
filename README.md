# Deep Convolutional Neural Network for Vocal Isolation from Music

This repository provides a configurable deep convolutional neural network to isolate vocals from music written in [python3.6](https://docs.python.org/3/). It is based on the [acapellabot](https://github.com/madebyollin/acapellabot) by [madebyollin](https://github.com/madebyollin).
To train the network the [MedleyDB](http://medleydb.weebly.com/) dataset was used.

## Dependencies
The following libraries and packages need to be installed to use this project.

### Training and Inference

* [tensorflow](https://www.tensorflow.org/) (Python library)
* [keras](https://keras.io/) (Python library)
* [librosa](http://librosa.github.io/librosa/) (Python library)
* [h5py](https://www.h5py.org/) (Python library)
* [numpy](http://www.numpy.org/) (Python library)


### Analysis

* [pydot](https://pypi.org/project/pydot/) (Package)
* [graphviz](https://graphviz.readthedocs.io/en/stable/) (Package)
* [oct2py](https://pypi.org/project/oct2py/) (Python library)
* [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) (Python library)
* [python3-tk](https://docs.python.org/3/library/tk.html) (Package)
* [octave](https://www.gnu.org/software/octave/) (Package)
* [octave-signal](https://wiki.octave.org/Signal_package) (Package)

## Project Structure

* **analysis.py** run different analysis
functionalities
* **batch.py** different batch generators for training
* **checkpointer.py** checkpoint hooks for keras
* **chopper.py** different slicing functions to create samples
* **config.py** configuration object reading environment variables
* **console.py** class for logging
* **conversion.py** utility to convert audio files to spectrograms and back
* **data.py** generates data to train on
* **grid_search.py** performs grid search using .yml file configurations
* **loss.py** different loss functions for training
* **metrics.py** different metrics used for training
* **modeler.py** different models to be used for training
* **normalizer.py** different normalizer strategies for data preparation
* **optimizer.py** different optimizers to be used for training
* **stoi.m** matlab file to calculate the STOI
* **vocal_isolation.py** runs the project

## Configuration
The settings used for execution are configurable by either exporting the appropriate environment variables, by directly setting the values in the `Config` class or, when using the grid search, by specifying the configuration in a `.yml` file. The configuration is applied using reflection.</br>
Some predefined configurations can be found in the `envs` directory. Source the environment file to load the configuration.</br>
E.g. for the `lps` environment run
>`source envs/lps`

A list of available options can be found at the end of this readme.

## Training

To train the network the corpus needs to contain the following files for each of the tracks:

* Complete mixture, suffix: `_all`
* Vocals separately, suffix: `_vocal`
* Instrumentals separately, suffix: `_instrumental`

The `DATA` needs to point to the directory where the corpus is stored.</br>
The training split can be specified by `SPLIT`. The `Config` class also contains an option to define the validation and test tracks directly.
The amount of epochs to train for can be set using `EPOCHS`.</br>
The `WEIGHTS` points to the `.h5` or `.hdf5` file in which the weights should be stored.</br>
More configuration options can be found at the end of this readme.</br>
After the configuration the project can be executed by invoking

> `python3 vocal_isolation.py`

The execution logs can be found in the `LOG_BASE` directory.

### Learning approaches
Two different learning approaches are available. The first one is similar to the original acapellabot using the log-power spectrograms (LPS) to train on. In this approach the phase information is lost and needs to be reconstructed using successive approximation. </br>
The second approach uses the real and imaginary parts of the complex spectrograms (RI) to learn the phase as well.</br>
The learning approach can be chosen by setting `LEARN_PHASE` to `False` for LPS or to `True` for RI.

### Grid Search
To train multiple configurations in one execution the `grid_search.py` can be used. It reads `.yml` files stored in `grid-search-configs` folder and creates configurations for every possible combination in the `.yml` file. When using the grid search all the output artifacts, including the weights, will be written to subfolders per configuration in the `LOG_BASE` directory.</br>
The grid search can be executed by invoking

>`python3 grid_search.py [myconfig.yml]`.

If no `.yml` file is specified it will use the default `grid_search.yml` containing all possible configurations.

## Inference
After the network is trained the produced weights stored in `WEIGHTS` can be used to perform inference on a given track to isolate the vocals. As the inference is computationally expensive it is not performed on the complete file, but smaller slices. The size of such a slice can be set by `INFERENCE_SLICE`. If your computer runs out of memory while inferencing, consider reducing the slice size.</br>
An inference can be executed by invoking

>`python3 vocal_isolation.py filetoinfer.wav`


## Analysis
Different functionalities for analysis are available in the `Analysis` class including the short-term-objective intelligibility (STOI) measure using the [stoi.m](https://github.com/posenhuang/deeplearningsourceseparation/blob/master/codes/denoising/demo/stoi.m) to calculate how well the produced output is understandable.
The `analysis.py` can be invoked using the following parameters:

* `--analyse` or `-a` : Analysis method to be executed
* `--save` or `-s` : Specifies whether the result should be saved
* `*args` : additional arguments depending on the analysis functionality


If the save option is specified the results will be written to the directory given by `ANALYSIS_PATH`.</br>
The following analysis functionalities are available:

### STOI
Calculate the stoi value.
#### Arguments

* path to mix file
* [path to clean file]


If both arguments are given the specified files will be used for the STOI calculation. Otherwise the clean file will be determined using the mix file.

>`python3 analysis.py -a stoi -s myfile.wav [cleanfile.wav]`

### Percentile
Calculate the value distributions and their difference to the median for each percentile on the whole data set located at `DATA` and create a box plot.

#### Arguments
No additional arguments are required.

>`python3 analysis.py -a percentile -s`

### MSE
Calculate the mean squared error between a processed and clean vocal file.

#### Arguments

* [path to processed file]
* [path to clean vocal file]

If no arguments are given the MSE analysis calculates the mean squared error for each track in the validation and test set.

>`python3 analysis.py -a mse -s [myprocessedvocal.wav] [cleanvocal.wav]`

### Volume
Scales the volume between a ratio of 1/100 and 100 and calculates the MSE for each ratio and plots the result.

#### Arguments
* path to mix file

>`python3 -a volume -s myfile.wav`

### Distribution
Calculates the value distributions of the dataset and plots them in a histogram.

#### Arguments
No additional arguments required

>`python3 -a distribution -s`

## Available Configuration Options
| Variable | Description | Possible Values | Default |
|----------|-------------|-----------------|---------|
| ANALYSIS_PATH	| Path to store analysis results | valid directory | "./analysis"
| BATCH | Batch size used for training | number > 0 | "8"
| BATCH_GENERATOR | Batch generator used for sample creation	| keras, default, track, random	| "random"
| CHECKPOINTS | Checkpoints to be used by keras	| tensorboard, weights, early_stopping | "tensorboard,weights"
| CHOPNAME | Slicing function for sample creation | tile, full, sliding_full, filtered, filtered_full, random, random_full, infere (only used for inference) |"tile"
| CHOPPARAMS | Parameter to configure slicing function	| scale (sample size), step (for sliding*), slices (for random*), upper (only use low frequencies), filter (for filter*)	| "{'scale': 128, 'step': 64, 'slices':256, 'upper':False, 'filter':'maximum'}"
| DATA | Path to training data | valid directory 	| "../bot_data"
| EARLY_STOPPING | Parameters for early stopping checkpoint  | min_delta, patience | "{'min_delta': 0.001, 'patience': 3}"
| EPOCHS | Amount of epochs to train for | number > 0 | "10"
| EPOCH_STEPS | Amount of samples for the random generator | number > BATCH | "50000"
| FFT | Window size for STFT | number > 0 | "1536"
| INFERENCE_SLICE | Slice size for inference | number > 0 | "3500"
| INSTRUMENTAL | Flag to train on instrumentals | True, False | "False"
| LOAD | Flag to load previous weights | True, False | "False"
| LOG_BASE | Log directory | valid directory | "./logs"
| LOSS | The loss function to be used by keras | mean_squared_error, mean_absolute_error, mean_squared_log_error	| "mean_squared_error"
| METRICS | Metrics to be used by keras	|"mean_pred,max_pred" | "mean_pred,max_pred"
| MODEL | The model to be used for training | acapellabot, leaky_dropout | "leaky_dropout"
| MODEL_PARAMS | The parameters to configure the model (only leaky_dropout) | alpha1 and alpha 2 for leakyReLu, rate for dropout	| "{'alpha1': 0.1,'alpha2': 0.01,'rate': 0.1}"
| NORMALIZER | The normalizer for data preparation | dummy (no normalization), percentile | "percentile"
| NORMALIZER_PARAMS	| The parameters to configure the normalizer (only percentile) | percentile	| "{'percentile': 99}"
| OPTIMIZER | The optimizer to be used by keras	| adam, rmsprop	| "adam"
| OPTIMIZER_PARAMS | The parameters to configure the optimizer | - | ""
| PHASE_ITERATIONS | The amount of iterations for the phase reconstruction | number > 0 | "10"
| LEARN_PHASE | Flag to perform LPS or RI learning | True (RI), False (LPS) | "True"
| QUIT | Flag to quit after training | False, True |"True"
| SPLIT | Percentage for training / validation split | float between 0 and 1 | "0.9"
| START_EPOCH | Starting epoch | number >= 0 | "0"
| TENSORBOARD | Directory to store tensorboard output | valid directory | "./tensorboard"
| TENSORBOARD_INFO | Amount of information to be returned | full, default | "default"
| WEIGHTS | Path to weight file | .h5 or .hdf5 file | "weights/weights.h5"
