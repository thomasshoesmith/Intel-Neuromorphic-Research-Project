#export CUDA_PATH=/usr/local/cuda
import numpy as np
import csv
from tqdm import trange
import os
import pickle
import json
from datetime import datetime

from ml_genn import Network, Population, Connection
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback, OptimiserParamSchedule
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput, LeakyIntegrateFireInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from time import perf_counter

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

import nvsmi
import json
import opendatasets as od

from ml_genn.compilers.event_prop_compiler import default_params

# Kaggle dataset directory
dataset = 'https://www.kaggle.com/datasets/thomasshoesmith/spiking-google-speech-commands/data'

# Using opendatasets to download SGSC dataset
od.download(dataset)

x_train = np.load("spiking-google-speech-commands/training_x_spikes.npy", allow_pickle=True)
y_train = np.load("spiking-google-speech-commands/training_y_spikes.npy", allow_pickle=True)

x_test = np.load("spiking-google-speech-commands/testing_x_spikes.npy", allow_pickle=True)
y_test = np.load("spiking-google-speech-commands/testing_y_spikes.npy", allow_pickle=True)

x_validation = np.load("spiking-google-speech-commands/validation_x_spikes.npy", allow_pickle=True)
y_validation = np.load("spiking-google-speech-commands/validation_y_spikes.npy", allow_pickle=True)


with open("SC_params.json", "r") as f: 
    params = json.load(f)

schedule_epoch_total = 0

# READ TO BE IMPLEMENTED 
# Preprocess
x_train_spikes = []
for i in range(len(x_train)):
    events = x_train[i]
    x_train_spikes.append(preprocess_tonic_spikes(events, 
                                                  x_train[0].dtype.names,
                                                  (80, 1, 1),
                                                  time_scale = 1))

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(x_train_spikes)
latest_spike_time = calc_latest_spike_time(x_train_spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Preprocess
x_validation_spikes = []
for i in range(len(x_validation)):
    events = x_validation[i]
    x_validation_spikes.append(preprocess_tonic_spikes(events, 
                                                       x_validation[0].dtype.names, 
                                                       (80, 1, 1),
                                                       time_scale = 1))

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = 80 #int(np.prod(x.sensor_size))
num_output = 35 #len(x.classes)


os.chdir("output")

# change dir for readout files
# lazy fix until a solution can be implemented with ml_genn to support output file directory change
try:
    os.makedirs(params.get("output_dir") + params.get("sweeping_suffix"))
except:
    pass

os.chdir(params.get("output_dir") + params.get("sweeping_suffix"))

# readout class
class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop, resume):
        # Create CSV writer
        self.file = open(filename, "a" if resume else "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row if we're not resuming from an existing training run
        if not resume:
            self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "accuracy", "Time", "memory"])

        self.output_pop = output_pop

    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        processes = nvsmi.get_gpu_processes()
        process = next(p for p in processes if p.pid == os.getpid())
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, 
                                m.total, 
                                m.correct,
                                m.correct / m.total,
                                perf_counter() - self.start_time,
                                process.used_memory])
        self.file.flush()
        
# Create sequential model
serialiser = Numpy(params.get("model_description"))
network = Network(default_params)

with network:
    # Populations
    """
    input = Population(LeakyIntegrateFireInput(v_thresh=1, 
                                            tau_mem=20,    
                                            input_frames=params.get("NUM_FRAMES"), 
                                            input_frame_timesteps=params.get("INPUT_FRAME_TIMESTEP")),
                        params.get("NUM_INPUT"), 
                        record_spikes = True)
    """
    
    input = Population(SpikeInput(max_spikes = params["BATCH_SIZE"] * max_spikes),
                       num_input,
                       record_spikes=True)
    
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, 
                                    tau_mem=20.0),
                params.get("NUM_HIDDEN"), 
                record_spikes=True)
    
    output = Population(LeakyIntegrate(tau_mem=20.0, 
                                readout="avg_var_exp_weight"),
                params.get("NUM_OUTPUT"), 
                record_spikes=True)

    # Connections
    Connection(input, hidden, Dense(Normal(mean = params.get("input_hidden_w_mean"), 
                                            sd = params.get("input_hidden_w_sd"))),
                Exponential(2.0))
    
    if params.get("recurrent"):
        Connection(hidden, hidden, Dense(Normal(mean = params.get("hidden_hidden_w_mean"), 
                                                sd = params.get("hidden_hidden_w_sd"))),
                Exponential(2.0))
    
    Connection(hidden, output, Dense(Normal(mean = params.get("hidden_output_w_mean"),
                                sd = params.get("hidden_output_w_sd"))),
                Exponential(2.0))
    
compiler = EventPropCompiler(example_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                        losses="sparse_categorical_crossentropy",
                        optimiser=Adam(params.get("lr")), 
                        batch_size = params.get("BATCH_SIZE"),
                        reg_lambda_lower = params.get("reg_lambda_lower"),
                        reg_lambda_upper = params.get("reg_lambda_upper"),
                        reg_nu_upper = params.get("reg_nu_upper"),
                        dt = params.get("dt"),
                        max_spikes=max_spikes)
                        #selectGPUByDeviceID=True, 
                        #deviceSelectMethod=DeviceSelect_MANUAL)

compiled_net = compiler.compile(network)

with compiled_net:

    # save parameters for reference
    json_object = json.dumps(params, indent = 4)
    with open("params.json", "w") as outfile:
        outfile.write(json_object)

    # Evaluate model on numpy dataset
    start_time = perf_counter() 
    
    # main dictionaries for tracking data # TODO: fix so debugging can be switched off
    metrics, metrics_val, cb_data_training, cb_data_validation = {}, {}, {}, {}
    
    if params.get("debug"):
        cb_data_training["input_spike_counts"] = []
        cb_data_validation["input_spike_counts"] = []

        cb_data_training["hidden_spike_counts"] = []
        cb_data_validation["hidden_spike_counts"] = []
        
    if params.get("record_all_hidden_spikes"):
        cb_data_training["hidden_spike_counts_unfiltered"] = []
        cb_data_validation["hidden_spike_counts_unfiltered"] = []
    
    # alpha decay after 1st epoch at a rate of 0.95
    def alpha_schedule(epoch, alpha):
        global schedule_epoch_total # TODO: remove this
        schedule_epoch_total = schedule_epoch_total + 1
        #print(schedule_epoch_total, params.get("lr_decay_rate") * 2, schedule_epoch_total % (params.get("lr_decay_rate") * 2) != 0)
        if params.get("lr_decay_rate") > 0 and schedule_epoch_total % (params.get("lr_decay_rate") * 2) != 0 and epoch != 0:
            return alpha * params.get("lr_decay")
        return alpha
                
    callbacks = [CSVTrainLog(f"train_output.csv", 
                        output,
                        False),
                Checkpoint(serialiser)]

    if params.get("recurrent"):
        callbacks.append(OptimiserParamSchedule("alpha", alpha_schedule))
        pass

    if params.get("verbose"):
        callbacks.append("batch_progress_bar")
        
    if params.get("debug"):
        callbacks.append(SpikeRecorder(input, 
                                key = "input_spike_counts", 
                                example_filter = 7))

        callbacks.append(SpikeRecorder(hidden, 
                                key = "hidden_spike_counts", 
                                example_filter = 7))
        
        callbacks.append(VarRecorder(output, 
                                        var = "v",
                                        example_filter = 7))
        
    if params.get("record_all_hidden_spikes"):
        callbacks.append(SpikeRecorder(hidden, 
                                key = "hidden_spike_counts_unfiltered", 
                                record_counts = True))

    metrics, metrics_val, cb_data_training, cb_data_validation = compiled_net.train({input: x_train_spikes},
                                                                                    {output: y_train},
                                                                                    num_epochs = params["NUM_EPOCH"],
                                                                                    shuffle = True,
                                                                                    callbacks = callbacks,
                                                                                    validation_x = {input: x_validation_spikes},
                                                                                    validation_y = {output: y_validation})  
    

    end_time = perf_counter()
    print(f"Time = {end_time - start_time}s")
        
    if params.get("debug"):
        # pickle serialisers
        with open('serialisers.pkl', 'wb') as f:
            pickle.dump(serialiser, f)

        # save input training spike counts # old way, can do this in 1 line?
        with open(f'input_training_spike_counts.npy', 'wb') as f:     
            input_spike_counts = np.array(cb_data_training["input_spike_counts"], dtype=np.int16)
            np.save(f, input_spike_counts)

        # save hidden training spike counts
        with open(f'hidden_training_spike_counts.npy', 'wb') as f:     
            hidden_spike_counts = np.array(cb_data_training["hidden_spike_counts"], dtype=np.int16)
            np.save(f, hidden_spike_counts)
            
        # get hidden spikes if param is true
    if params.get("record_all_hidden_spikes"):
        # save all hidden spike counts
        with open(f'hidden_spike_counts_unfiltered.npy', 'wb') as f:     
            hidden_spike_counts = np.array(cb_data_training["hidden_spike_counts_unfiltered"], dtype=np.int16)
            np.save(f, hidden_spike_counts)
        
# reset directory
os.chdir("..")

# if sweeping, save params and accuracy to csv
does_summary_file_exist = os.path.exists("summary.csv")
# Create CSV writer
print("Writing summary")
print(os.getcwd())
file = open("summary.csv", "a" if does_summary_file_exist else "w")
csv_writer = csv.writer(file, delimiter=",")

# Write header row if we're not resuming from an existing training run
if not does_summary_file_exist:
    csv_writer.writerow((list(params) + ["accuracy", "validation", "date"]))

csv_writer.writerow((list(params.values()) + \
                        [round(metrics[output].correct / metrics[output].total, 5),
                        round(metrics_val[output].correct / metrics_val[output].total, 5),
                        (str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))]))