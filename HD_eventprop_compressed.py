#export CUDA_PATH=/usr/local/cuda
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tqdm import trange
import os
import pickle

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

from HD_eventprop import hd_eventprop

# constants
params = {}
params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 20
params["BATCH_SIZE"] = 256
params["INPUT_FRAME_TIMESTEP"] = 2
params["INPUT_SCALE"] = 0.00099
params["NUM_EPOCH"] = 50
params["NUM_FRAMES"] = 80
params["verbose"] = False
params["debug"] = False
params["lr"] = 0.01
params["dt"] = 1

params["reg_lambda_lower"] = 0#1e-12
params["reg_lambda_upper"] = 0#1e-12
params["reg_nu_upper"] = 0#2

#weights
params["hidden_w_mean"] = 0.0 #0.5
params["hidden_w_sd"] = 3.5 #4.0
params["output_w_mean"] = 3.0
params["output_w_sd"] = 1.5 

accuracy = hd_eventprop(params, 
                        file_path = os.path.expanduser("~/data/rawHD/experimental_3/"),
                        output_dir = "HD_eventprop_compressed",
                        model_description = "compressed")

print(f"accuracy of the network is {accuracy * 100:.2f}%")