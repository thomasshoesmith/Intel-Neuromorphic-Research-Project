import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput, LeakyIntegrateFireInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from time import perf_counter

from ml_genn.utils.data import (calc_latest_spike_time, linear_latency_encode_data)

from ml_genn.compilers.event_prop_compiler import default_params

import random
import librosa

# constants

NUM_INPUT = 40
NUM_HIDDEN = 256
NUM_OUTPUT = 20

NUM_EPOCH = 50

BATCH_SIZE = 128

INPUT_FRAME_TIMESTEP = 2

INPUT_SCALE = 0.008

# Load testing data
x_train = np.load("/its/home/ts468/data/rawHD/experimental_2/training_x_data.npy")
y_train = np.load("/its/home/ts468/data/rawHD/experimental_2/training_y_data.npy")

x_test = np.load("/its/home/ts468/data/rawHD/experimental_2/testing_x_data.npy")
y_test = np.load("/its/home/ts468/data/rawHD/experimental_2/testing_y_data.npy")

training_details = pd.read_csv("/its/home/ts468/data/rawHD/experimental_2/training_details.csv")
testing_details = pd.read_csv("/its/home/ts468/data/rawHD/experimental_2/testing_details.csv")

training_images = np.swapaxes(x_train, 1, 2) 
testing_images = np.swapaxes(x_test, 1, 2) 

training_images = training_images + abs(np.floor(training_images.min()))
testing_images = testing_images + abs(np.floor(testing_images.min()))

training_labels = y_train
testing_labels = y_test