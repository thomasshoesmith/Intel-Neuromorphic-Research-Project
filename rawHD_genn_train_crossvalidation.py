#export CUDA_PATH=/usr/local/cuda
import numpy as np
import csv
from tqdm import trange
import os
import pickle
import json
from datetime import datetime
import pandas as pd
import copy
import matplotlib.pyplot as plt
import re

from ml_genn import Network, Population, Connection
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback, OptimiserParamSchedule
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
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

from rawHD_dataset_loader_padded_spikes import rawHD_Loader

# Kaggle dataset directory
dataset = 'https://www.kaggle.com/datasets/thomasshoesmith/spiking-google-speech-commands/data'

# Using opendatasets to download SGSC dataset
od.download(dataset)

with open("rawHD_params.json", "r") as f: 
    params = json.load(f)
    
params["num_samples"] = None
params["cross_validation"] = True
params["cross_validation_run_all"] = False # is this needed?
    
x_train, y_train, z_train, x_test, y_test, z_test, x_validation, y_validation, z_validation = rawHD_Loader(dir = os.getcwd() + params["dataset_directory"],
                                                                                                           num_samples=params["num_samples"],
                                                                                                           shuffle = False,
                                                                                                           shuffle_seed = 0,
                                                                                                           process_padded_spikes = False,
                                                                                                           validation_split = 0.0)

if params.get("cross_validation"):
    training_details = pd.read_csv(os.getcwd() + params.get("dataset_directory") + "training_details.csv")
    testing_details = pd.read_csv(os.getcwd() + params.get("dataset_directory") + "testing_details.csv")

schedule_epoch_total = 0

# READ TO BE IMPLEMENTED 
# Preprocess
x_train_spikes = []
for i in range(len(x_train)):
    events = x_train[i]
    x_train_spikes.append(preprocess_tonic_spikes(events, 
                                                  x_train[0].dtype.names,
                                                  (params["NUM_INPUT"], 1, 1),
                                                  time_scale = 1))

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(x_train_spikes)
latest_spike_time = 2000 #calc_latest_spike_time(x_train_spikes) #TODO: Fix
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Preprocess
x_validation_spikes = []
for i in range(len(x_validation)):
    events = x_validation[i]
    x_validation_spikes.append(preprocess_tonic_spikes(events, 
                                                       x_validation[0].dtype.names, 
                                                       (params["NUM_INPUT"], 1, 1),
                                                       time_scale = 1))

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
        
def CSVValidationLog(filename, epoch, metric, population):
    file = open(filename, "a" if e > 0 else "w")
    csv_writer = csv.writer(file, delimiter=",")
    
    if e == 0:
        csv_writer.writerow(["Epoch", "Num trials", "Number correct", "accuracy", "memory"])

    processes = nvsmi.get_gpu_processes()
    process = next(p for p in processes if p.pid == os.getpid())
    m = metric[population]
    
    csv_writer.writerow([epoch, 
                            m.total, 
                            m.correct,
                            m.correct / m.total,
                            process.used_memory])
    file.flush()

# Create sequential model
serialiser = Numpy(params.get("model_description"))
network = Network(default_params)

with network:
    # Populations
    input = Population(SpikeInput(max_spikes = params["BATCH_SIZE"] * max_spikes),
                       params["NUM_INPUT"],
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
    i2h = Connection(input, hidden, Dense(Normal(mean = params.get("input_hidden_w_mean"), 
                                            sd = params.get("input_hidden_w_sd"))),
                Exponential(5.0))
    
    if params.get("recurrent"):
        h2h = Connection(hidden, hidden, Dense(Normal(mean = params.get("hidden_hidden_w_mean"), 
                                                sd = params.get("hidden_hidden_w_sd"))),
                Exponential(5.0))
    
    h2o = Connection(hidden, output, Dense(Normal(mean = params.get("hidden_output_w_mean"),
                                sd = params.get("hidden_output_w_sd"))),
                Exponential(5.0))
    
clamp_weight_conns_dir = {i2h: (-10, 10), h2o: (-10, 10)}
if params["recurrent"] : clamp_weight_conns_dir = {i2h: (-10, 10), h2h: (-10, 10), h2o: (-10, 10)}

compiler = EventPropCompiler(example_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                        losses="sparse_categorical_crossentropy",
                        optimiser=Adam(params.get("lr")), 
                        batch_size = params.get("BATCH_SIZE"),
                        reg_lambda_lower = params.get("reg_lambda_lower"),
                        reg_lambda_upper = params.get("reg_lambda_upper"),
                        reg_nu_upper = params.get("reg_nu_upper"),
                        dt = params.get("dt"),
                        max_spikes=max_spikes)
                        #clamp_weight_conns=clamp_weight_conns_dir)

compiled_net = compiler.compile(network)

if params.get("verbose"): print(training_details.head())
speaker_id = np.sort(training_details.Speaker.unique())
speaker_id = speaker_id = speaker_id.astype('int8')  #np where is fussy with int
if params.get("verbose"): print(np.sort(testing_details.Speaker.unique()))

# Create sequential model
serialisers = []
for s in speaker_id:
    serialisers.append(Numpy(f"serialiser_{s}"))

speaker = list(training_details.loc[:, "Speaker"])

# Evaluate model on numpy dataset
start_time = perf_counter() 

for count, speaker_left in enumerate(speaker_id):
    #reset gpu memory
    print(f"speaker {speaker_left} of {len(speaker_id)}")

    train_spikes, train_labels = [],[]
    eval_spikes, eval_labels = [],[]
    for i in np.where(speaker != speaker_left)[0]:
        train_spikes.append(x_train_spikes[i])
        train_labels.append(y_train[i])
    for i in np.where(speaker == speaker_left)[0]:
        eval_spikes.append(x_train_spikes[i])
        eval_labels.append(y_train[i])
    
    with compiled_net:

        # save parameters for reference
        json_object = json.dumps(params, indent = 4)
        with open("params.json", "w") as outfile:
            outfile.write(json_object)
        
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
        
        # alpha decay after 1st epoch at a rate of lr_decay_rate
        def alpha_schedule(epoch, alpha):
            global schedule_epoch_total # TODO: remove this
            schedule_epoch_total = schedule_epoch_total + 1
            #print(schedule_epoch_total, params.get("lr_decay_rate") * 2, schedule_epoch_total % (params.get("lr_decay_rate") * 2) != 0)
            if params.get("lr_decay_rate") > 0 and schedule_epoch_total % (params.get("lr_decay_rate") * 2) != 0 and epoch != 0:
                return alpha * params.get("lr_decay")
            return alpha
    
        # Evaluate model on numpy dataset
        callbacks = [Checkpoint(serialisers[count]), 
                     CSVTrainLog(f"train_output_{speaker_left}.csv", 
                                 output,
                                 False)]

        if params.get("recurrent"):
            callbacks.append(OptimiserParamSchedule("alpha", alpha_schedule))

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
            
        for e in trange(params["NUM_EPOCH"]):
            e_train_spikes = copy.deepcopy(train_spikes)
            metrics, metrics_val, t_cb_data_training, t_cb_data_validation = compiled_net.train({input: e_train_spikes},
                                                                                                {output: train_labels},
                                                                                                start_epoch = e,
                                                                                                num_epochs = 1,
                                                                                                shuffle = True,
                                                                                                callbacks = callbacks,
                                                                                                validation_x = {input: eval_spikes},
                                                                                                validation_y = {output: eval_labels})  
            
            CSVValidationLog(f"validation_output_{speaker_left}.csv", e, metrics_val, output)
                
            for key in list(cb_data_training.keys()):
                cb_data_training[key].append(t_cb_data_training[key])
                #cb_data_validation[key].append(t_cb_data_validation[key])

            # breaking out early if network is under performing (<10%)
            if metrics[output].correct / metrics[output].total < .1:
                print("exiting early due to collapsed network / poor performance")
                break

    end_time = perf_counter()
    print(f"Time = {end_time - start_time}s")
    
if params["cross_validation_run_all"]: 
    print("\n\nrun across all values ")
    combined_serialiser = Numpy("serialiser_all")

    with compiled_net:
        # Evaluate model on numpy dataset
        callbacks = [Checkpoint(combined_serialiser), 
                        CSVTrainLog(f"train_output_combined.csv", 
                                    output,
                                    False)]
        
        if params.get("verbose"):
            callbacks.append("batch_progress_bar")
        
        if params.get("debug"):
            print("!!!    debug")
            callbacks.append(SpikeRecorder(hidden, 
                                    key = "hidden_spike_counts", 
                                    record_counts = True,
                                    example_filter = 70))

        if params.get("record_all_hidden_spikes"):
            callbacks.append(SpikeRecorder(hidden, 
                                    key = "hidden_spike_counts_record", 
                                    record_counts = True))
            
        metrics, cb_data_training = compiled_net.train({input: x_train_spikes},
                                                        {output: y_train},
                                                        num_epochs = params.get("NUM_EPOCH"), 
                                                        shuffle = not(params.get("debug")),
                                                        callbacks=callbacks)
        
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
        
if params["cross_validation"]:
    train_files, validation_files = [], []
    for f in os.listdir():
        if f.startswith("train_output"):
            train_files.append(f)
        if f.startswith("validation_output"):
            validation_files.append(f)

    assert len(train_files) == len(validation_files), "mismatch of val to train output"

    train_accuracy = {}
    for file in train_files:
        df = pd.read_csv(file)
        train_accuracy[int(re.split(r'[_\.]', file)[2])] = df.loc[df.index[-1], 'accuracy']

    validation_accuracy = {}
    for file in validation_files:
        df = pd.read_csv(file)
        validation_accuracy[int(re.split(r'[_\.]', file)[2])] = df.loc[df.index[-1], 'accuracy']

    speaker = list(dict(sorted(train_accuracy.items())).keys())
    training = list(dict(sorted(train_accuracy.items())).values())
    validation = list(dict(sorted(validation_accuracy.items())).values())

    # X-axis positions
    x = np.arange(len(speaker))
    width = 0.35

    plt.bar(x - width/2, training, width, label='train')
    plt.bar(x + width/2, validation, width, label='validation')

    plt.xlabel('Speaker ID')
    plt.ylabel('Accuracy')
    plt.title('Cross Validation Accuracy for rawHD')
    plt.xticks(x, speaker)

    plt.axhline(sum(train_accuracy.values()) / len(train_accuracy), color = "C0", linestyle = (2, (1, 1)), label = f"training accuracy avg ({np.round(sum(train_accuracy.values()) / len(train_accuracy), 4)})")
    plt.axhline(sum(validation_accuracy.values()) / len(validation_accuracy), color = "C1", linestyle = (2, (1, 1)), label = f"validation accuracy avg ({np.round(sum(validation_accuracy.values()) / len(validation_accuracy), 4)})")

    plt.tight_layout()
    plt.legend()
    plt.savefig("cross_val_plot.png")
        
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