########################################################
# * converts a current injection dataset to spike times
# * script useful for custom dataset configs
# * currently requires 80GB system memory for DB loading
########################################################

import numpy as np
from tqdm import trange
import os

params = {}
params["to_convert_directory"] = "/its/home/ts468/data/rawHD/rawHD_80input_updated/" #"/its/home/ts468/data/rawSC/rawSC_40input_updated/"
params["output_directory"] = "spiking-heidleberg-digits-80input" # "spiking-google-speech-commands-40input"
params["num_frames"] = 2
params["shuffle"] = False
params["num_samples"] = None
params["tau_mem"] = 20.0
params["dt_ms"] = 1.0
params["vth"] = 1.0
params["to_concatenate"] = False

# Load dataset
def load_gsc(params):
    train_x = np.load(os.path.expanduser(params["to_convert_directory"]) + "training_x_data.npy")
    train_y = np.load(os.path.expanduser(params["to_convert_directory"]) + "training_y_data.npy")
    
    test_x = np.load(os.path.expanduser(params["to_convert_directory"]) + "testing_x_data.npy")
    test_y = np.load(os.path.expanduser(params["to_convert_directory"]) + "testing_y_data.npy")
   
    # adding validation data if exists
    validation_x = np.array([])
    validation_y = np.array([])
    if os.path.isfile(os.path.expanduser(params["to_convert_directory"]) + "validation_y_data.npy"):
        print("!! validation dataset loaded successfully !!")
        validation_x = np.load(os.path.expanduser(params["to_convert_directory"]) + "validation_x_data.npy")
        validation_y = np.load(os.path.expanduser(params["to_convert_directory"]) + "validation_y_data.npy")

    # shuffle array before stacking (stacking like ring buffer) 
    if params["shuffle"]:
        shuffler = np.random.permutation(len(train_x))
        train_x = train_x[shuffler]
        train_y = train_y[shuffler]

        # unnecessary, but being used for debugging 
        shuffler = np.random.permutation(len(train_x))
        train_x = train_x[shuffler]
        train_y = train_y[shuffler]

    # crop the num of samples
    train_x, train_y = train_x[:params["num_samples"]], train_y[:params["num_samples"]]
    validation_x, validation_y = validation_x[:params["num_samples"]], validation_y[:params["num_samples"]]
    test_x, test_y = test_x[:params["num_samples"]], test_y[:params["num_samples"]]

    # repeat frames for current injection over (frame) time
    train_x = np.repeat(train_x, params["num_frames"], axis = 1)
    validation_x = np.repeat(validation_x, params["num_frames"], axis = 1)
    test_x = np.repeat(test_x, params["num_frames"], axis = 1)

    # to concatenate
    if params["to_concatenate"]:
        train_x = np.concatenate(train_x, axis = 0)
        validation_x = np.concatenate(validation_x, axis = 0)
        test_x = np.concatenate(test_x, axis = 0)

    return train_x, train_y, validation_x, validation_y, test_x, test_y

# load current injection gsc dataset
train_x, train_y, validation_x, validation_y, test_x, test_y = load_gsc(params)

def convert_currents_to_spike_times(params, this_x):
    alpha = np.exp(-params["dt_ms"] / params["tau_mem"])

    number_of_trials = this_x.shape[0]      
    number_of_frames = this_x.shape[1]
    number_of_channels = this_x.shape[2] 

    spike_times = []

    for trial in trange(number_of_trials):

        v = np.zeros((number_of_channels))
        spike_times_per_trial = []

        for timestep in range(number_of_frames):
            v[:] = (v * alpha) + this_x[trial][timestep]
            spikes_out = v > params["vth"]
            v[spikes_out] = 0

            spike_indexes = np.where(spikes_out == True)[0]

            for spike_index in spike_indexes:
                spike_times_per_trial.append((spike_index, timestep, 1))

        spike_times.append(np.array(spike_times_per_trial, 
                                    dtype = ([('x', np.int8), 
                                              ('t', np.uint16),
                                              ('p', np.int8)])))
        
    return np.array(spike_times, dtype = object)

try:
    os.makedirs(params["output_directory"])
except:
    pass

print("converting training dataset")
np.save(f"{params['output_directory']}/training_x_spikes.npy", convert_currents_to_spike_times(params, train_x))
np.save(f"{params['output_directory']}/training_y_spikes.npy", np.array(train_y, dtype = np.uint8))

print("converting testing dataset")
np.save(f"{params['output_directory']}/testing_x_spikes.npy", convert_currents_to_spike_times(params, test_x))
np.save(f"{params['output_directory']}/testing_y_spikes.npy", np.array(test_y, dtype = np.uint8))

if os.path.isfile(os.path.expanduser(params["to_convert_directory"]) + "validation_y_data.npy"):
    print("converting validation dataset")
    np.save(f"{params['output_directory']}/validation_x_spikes.npy", convert_currents_to_spike_times(params, validation_x))
    np.save(f"{params['output_directory']}/validation_y_spikes.npy", np.array(validation_y, dtype = np.uint8))