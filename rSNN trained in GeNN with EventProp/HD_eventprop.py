import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os

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

# readout parameters

verbose = True

# change dir for readout files

try:
    os.mkdir("rSNN trained in GeNN with EventProp/HD_eventprop")
except:
    pass

os.chdir("rSNN trained in GeNN with EventProp/HD_eventprop")

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

if verbose: print(testing_details.head())
speaker_id = np.sort(testing_details.Speaker.unique())
if verbose: print(np.sort(testing_details.Speaker.unique()))

# readout class

class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop, resume):
        # Create CSV writer
        self.file = open(filename, "a" if resume else "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row if we're not resuming from an existing training run
        if not resume:
            self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "accuracy", "Time"])

        self.output_pop = output_pop

    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, 
                                  m.total, 
                                  m.correct,
                                  m.correct / m.total,
                                  perf_counter() - self.start_time])
        self.file.flush()
        
# Create sequential model
serialiser = Numpy("latency_hd_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(LeakyIntegrateFireInput(v_thresh=4,
                                               tau_mem=10, 
                                               input_frames=80, 
                                               input_frame_timesteps=INPUT_FRAME_TIMESTEP),
                        NUM_INPUT, 
                        record_spikes = True)
    
    hidden = Layer(Dense(Normal(mean=0.5, # m = .5, sd = 4 ~ 68%
                                sd=4.0)), 
                   LeakyIntegrateFire(v_thresh=5.0, 
                                      tau_mem=20.0,
                                      tau_refrac=None),
                   NUM_HIDDEN, 
                   Exponential(5.0), #5
                   record_spikes=True)
    
    output = Layer(Dense(Normal(mean=0.5, # m = 0.5, sd = 1 @ ~ 66
                                sd=1)),
                   LeakyIntegrate(tau_mem=20.0, 
                                  readout="avg_var"),
                   NUM_OUTPUT, 
                   Exponential(5.0), #5
                   record_spikes=True)
    
compiler = EventPropCompiler(example_timesteps=80 * INPUT_FRAME_TIMESTEP,
                         losses="sparse_categorical_crossentropy",
                         optimiser=Adam(0.01), batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    start_epoch = 0
    callbacks = ["batch_progress_bar", 
                 Checkpoint(serialiser), 
                 CSVTrainLog("train_output.csv", 
                             output,
                             False)]
    metrics  = compiled_net.train({input: training_images * INPUT_SCALE},
                                      {output: training_labels},
                                      num_epochs=NUM_EPOCH, 
                                      shuffle=True,
                                      validation_split=0.1,
                                      callbacks=callbacks)
    
    
# evaluate

network.load((NUM_EPOCH - 1,), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=80 * INPUT_FRAME_TIMESTEP,
                             reset_in_syn_between_batches=True,
                             batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:
    callbacks = ["batch_progress_bar", 
                 SpikeRecorder(input, key="input_spikes"), 
                 SpikeRecorder(hidden, key="hidden_spikes"),
                 SpikeRecorder(output, key="output_spikes"),
                 VarRecorder(output, "v", key="v_output")]
    metrics, cb_data = compiled_net.evaluate({input: training_images * INPUT_SCALE},
                                             {output: training_labels},
                                             callbacks=callbacks)
    
if verbose:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('rawHD with EventProp on ml_genn')

    value = random.randint(0, len(x_test))

    ax1.scatter(cb_data["hidden_spikes"][0][value], 
                cb_data["hidden_spikes"][1][value], s=1)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Neuron ID")
    ax1.set_title("Hidden")
    ax1.set_xlim(0, 80 * INPUT_FRAME_TIMESTEP)
    ax1.set_ylim(0, 40)

    ax2.scatter(cb_data["input_spikes"][0][value], 
                cb_data["input_spikes"][1][value], s=1)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Neuron ID")
    ax2.set_title("Input")
    ax2.set_xlim(0, 80 * INPUT_FRAME_TIMESTEP)
    ax2.set_ylim(0, 40)

    ax3.plot(cb_data["v_output"][value])
    ax3.set_xlabel("Time [ms]")
    ax3.set_ylabel("voltage (v)")
    ax3.set_title("Output voltage")
    ax3.set_xlim(0, 80 * INPUT_FRAME_TIMESTEP)
    #ax3.set_ylim(0, 40)

    sr = 22050
    img = librosa.display.specshow(x_train[value], 
                            x_axis='time', 
                            y_axis='mel', 
                            sr=sr, 
                            cmap='viridis')
    #fig.colorbar(img, ax = ax4)
    ax4.set_title("mel encoding")

    fig.tight_layout()

    plt.show()
    
    data = pd.read_csv("/its/home/ts468/PhD/Intel-Neuromorphic-Research-Project/rSNN trained in GeNN with EventProp/train_output.csv")
    df = pd.DataFrame(data, columns=['accuracy'])

    accuracy = np.array(df)

    accuracy = accuracy * 100

    validation = []
    training = []

    for i in range(len(accuracy)):
        if i % 2 == 0:
            training.append(float(accuracy[i]))
        else:
            validation.append(float(accuracy[i]))
            
            
    plt.plot(training, label = "training")
    plt.plot(validation, label = "validation")
    plt.ylabel("accuracy (%)")
    plt.xlabel("epochs")
    plt.title("accuracy during training")
    plt.legend()
    plt.show()