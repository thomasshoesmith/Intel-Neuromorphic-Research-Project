#export CUDA_PATH=/usr/local/cuda
# sudo apt-get install python3-tk
import numpy as np
#import matplotlib
#import tkinter
#matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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

# constants
params = {}
params["NUM_INPUT"] = 40
params["NUM_HIDDEN"] = 256
params["NUM_OUTPUT"] = 20
params["BATCH_SIZE"] = 128
params["INPUT_FRAME_TIMESTEP"] = 2
params["INPUT_SCALE"] = 0.008
params["NUM_EPOCH"] = 50
params["NUM_FRAMES"] = 80
params["verbose"] = False
params["lr"] = 0.01

#weights
params["hidden_w_mean"] = 0.0 #0.5
params["hidden_w_sd"] = 3.5 #4.0
params["output_w_mean"] = 3.0 #0.5
params["output_w_sd"] = 1.5 #1

file_path = os.path.expanduser("~/data/rawHD/experimental_2/")#"/its/home/ts468/data/rawHD/experimental_2/"#"/home/ts468/Documents/data/rawHD/experimental_2/"


def hd_eventprop(params, file_path, return_accuracy = True):
    """
    Function to run hd classification using eventprop
    Parameters:
      params - a dictionary containing all parameters
      file_path - directory where training/testing/detail files are found
      return_accuracy - bool for if cvs train log is generated, or an accuracy returned
    """

    # change dir for readout files
    try:
        os.mkdir("HD_eventprop_cross_validation_output")
    except:
        pass

    os.chdir("HD_eventprop_cross_validation_output")

    # Load testing data
    x_train = np.load(file_path + "training_x_data.npy")
    y_train = np.load(file_path + "training_y_data.npy")

    x_test = np.load(file_path + "testing_x_data.npy")
    y_test = np.load(file_path + "testing_y_data.npy")

    training_details = pd.read_csv(file_path + "training_details.csv")
    testing_details = pd.read_csv(file_path + "testing_details.csv")

    training_images = np.swapaxes(x_train, 1, 2) 
    testing_images = np.swapaxes(x_test, 1, 2) 

    training_images = training_images + abs(np.floor(training_images.min()))
    testing_images = testing_images + abs(np.floor(testing_images.min()))

    training_labels = y_train
    testing_labels = y_test

    speaker_id = np.sort(training_details.Speaker.unique())
    
    speaker = list(training_details.loc[:, "Speaker"])

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
        
    network = SequentialNetwork(default_params)
    with network:
        # Populations
        input = InputLayer(LeakyIntegrateFireInput(v_thresh=4,
                                                tau_mem=10, 
                                                input_frames=params.get("NUM_FRAMES"), 
                                                input_frame_timesteps=params.get("INPUT_FRAME_TIMESTEP")),
                            params.get("NUM_INPUT"), 
                            record_spikes = True)
        
        hidden = Layer(Dense(Normal(mean = params.get("hidden_w_mean"), # m = .5, sd = 4 ~ 68%
                                    sd = params.get("hidden_w_sd"))), 
                    LeakyIntegrateFire(v_thresh=5.0, 
                                        tau_mem=20.0,
                                        tau_refrac=None),
                    params.get("NUM_HIDDEN"), 
                    Exponential(5.0), #5
                    record_spikes=True)
        
        output = Layer(Dense(Normal(mean = params.get("output_w_mean"), # m = 0.5, sd = 1 @ ~ 66
                                    sd = params.get("output_w_sd"))),
                    LeakyIntegrate(tau_mem=20.0, 
                                    readout="avg_var"),
                    params.get("NUM_OUTPUT"), 
                    Exponential(5.0), #5
                    record_spikes=True)
        
    # pickle serialisers
    with open('serialisers.pkl', 'rb') as f:
        serialisers = pickle.load(f)
        
    # evaluate
    network.load((params.get("NUM_EPOCH") - 1,), serialisers[-1])

    compiler = InferenceCompiler(evaluate_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                                reset_in_syn_between_batches=True,
                                batch_size = params.get("BATCH_SIZE"))
    compiled_net = compiler.compile(network)

    with compiled_net:
        if return_accuracy:
            callbacks = [Checkpoint(serialisers[-1])]
        else:
            callbacks = ["batch_progress_bar", 
                        SpikeRecorder(input, key="input_spikes"), 
                        SpikeRecorder(hidden, key="hidden_spikes"),
                        SpikeRecorder(output, key="output_spikes"),
                        VarRecorder(output, "v", key="v_output"),
                        SpikeRecorder(hidden, key = "hidden_spike_counts", record_counts = True)]
    
        metrics, cb_data = compiled_net.evaluate({input: training_images * params.get("INPUT_SCALE")},
                                                {output: training_labels},
                                                callbacks = callbacks)
        
    if params.get("verbose") and not return_accuracy:
        # cannot print verbose whilst requesting just accuracy
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('rawHD with EventProp on ml_genn')

        value = random.randint(0, len(x_train))

        ax1.scatter(cb_data["hidden_spikes"][0][value], 
                    cb_data["hidden_spikes"][1][value], s=1)
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Neuron ID")
        ax1.set_title("Hidden")
        ax1.set_xlim(0, params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"))
        ax1.set_ylim(0, params.get("NUM_HIDDEN"))

        ax2.scatter(cb_data["input_spikes"][0][value], 
                    cb_data["input_spikes"][1][value], s=1)
        ax2.set_xlabel("Time [ms]")
        ax2.set_ylabel("Neuron ID")
        ax2.set_title("Input")
        ax2.set_xlim(0, params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"))
        ax2.set_ylim(0, params.get("NUM_INPUT"))

        ax3.plot(cb_data["v_output"][value])
        ax3.set_xlabel("Time [ms]")
        ax3.set_ylabel("voltage (v)")
        ax3.set_title("Output voltage")
        ax3.set_xlim(0, params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"))
        #ax3.set_ylim(0, params.get("NUM_INPUT"))

        #sr = 22050
        #img = librosa.display.specshow(x_train[value], x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
        plt.imshow(x_train[value], origin = "lower")
        ax4.set_ylabel("Neuron ID")
        ax4.set_xlabel("frames")
        ax4.set_xlim(0, params.get("NUM_FRAMES"))
        ax4.set_ylim(0, params.get("NUM_INPUT"))

        #fig.colorbar(img, ax = ax4)
        ax4.set_title("Input mel encoding")

        fig.tight_layout()

        plt.show()

        hidden_spike_counts = cb_data["hidden_spike_counts"]
        hidden_spikes = cb_data["hidden_spikes"]

        # Assert that manually-calculated spike counts from arbitrary example match those calculated using new system
        assert np.array_equal(np.bincount(hidden_spikes[1][value], 
                                          minlength=params.get("NUM_HIDDEN")),
                                          hidden_spike_counts[value])
        
        
        # monitoring spikes in hidden layer
        total_spikes = np.zeros(params.get("NUM_HIDDEN"))
        for i in range(len(hidden_spike_counts)):
            total_spikes = np.add(total_spikes, hidden_spike_counts[i])
        """

        plt.bar(list(range(len(total_spikes))), total_spikes)
        plt.title("Hidden Spikes per neuron across trail")
        plt.ylabel("total number of spikes across trial")
        plt.xlabel("Neuron ID")
        plt.show()
        """

        #print(np.sort(total_spikes))

        plt.bar(list(range(len(total_spikes))), np.sort(total_spikes))
        plt.title("Sorted hidden layer Spikes with {} silent neurons out of {} neurons".format(np.count_nonzero(total_spikes == 0), params.get("NUM_HIDDEN")))
        plt.ylabel("total number of spikes across trial")
        plt.xticks([])
        plt.xlim(0, params.get("NUM_HIDDEN"))
        plt.show()

        print(f"number of silent neurons: {np.count_nonzero(total_spikes == 0)}")
        print("total", len(hidden_spike_counts))

        #figure(figsize=(8, 6), dpi=200)
        # show accuracy log
        for speaker_left in speaker_id:
    
            data = pd.read_csv(f"train_output_{speaker_left}.csv")
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
                    
                    
            plt.plot(training, label = f"training speaker {speaker_left}")
            plt.plot(validation, label = f"validation speaker {speaker_left}")
        plt.ylabel("accuracy (%)")
        plt.xlabel("epochs")
        plt.title("accuracy of training data during training")
        plt.ylim(0, 90)
        plt.legend()
        plt.show()
    
    # reset directory
    os.chdir("..")
    
    if return_accuracy:
        return metrics[output].correct / metrics[output].total


params["verbose"] = True
get_accuracy = False
accuracy = hd_eventprop(params, file_path, get_accuracy)

if get_accuracy: print(F"accuracy of the network is {accuracy * 100:.2f}%")