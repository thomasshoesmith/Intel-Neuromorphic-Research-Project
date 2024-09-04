###############################################
#               Dataset loader                #
#       to a numpy array of fixed memory      #
###############################################

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange

dataset_directory = "/its/home/ts468/PhD/SGSC_Dataset/spiking-google-speech-commands/"
number_of_neurons = 80
number_of_timesteps = 2000

def SGSC_Loader(dir, 
                num_samples = None,
                shuffle = False,
                number_of_neurons = 80,
                number_of_timesteps = 2000):
    
    # loading x spike times
    x_train_spike_times = np.load(os.path.expanduser(dir) + "training_x_spikes.npy", allow_pickle = True)
    x_test_spike_times = np.load(os.path.expanduser(dir) + "testing_x_spikes.npy", allow_pickle = True)
    x_validation_spike_times = np.load(os.path.expanduser(dir) + "validation_x_spikes.npy", allow_pickle = True)

    # loading y labels to returning
    y_train = np.load(os.path.expanduser(dir) + "training_y_spikes.npy", allow_pickle = True)
    y_test = np.load(os.path.expanduser(dir) + "testing_y_spikes.npy", allow_pickle = True)
    y_validation = np.load(os.path.expanduser(dir) + "validation_y_spikes.npy", allow_pickle = True)

    # define the empty zeroed arrays
    x_train = np.zeros((x_train_spike_times.shape[0], number_of_neurons, number_of_timesteps), dtype = np.int8)
    x_test = np.zeros((x_test_spike_times.shape[0], number_of_neurons, number_of_timesteps), dtype = np.int8)
    x_validation = np.zeros((x_validation_spike_times.shape[0], number_of_neurons, number_of_timesteps), dtype = np.int8)

    # shuffle array
    if shuffle:
        shuffler = np.random.permutation(len(x_train))
        x_train = x_train[shuffler]
        y_train = y_train[shuffler]

        # unnecessary, but being used for debugging 
        shuffler = np.random.permutation(len(x_test))
        x_test = x_test[shuffler]
        y_test = y_test[shuffler]

    # crop the num of samples
    x_train, y_train = x_train[:num_samples], y_train[:num_samples]
    x_validation, y_validation = x_validation[:num_samples], y_validation[:num_samples]
    x_test, y_test = x_test[:num_samples], y_test[:num_samples]

    # flipping bits
    print("loading training")
    for trial in trange(x_train.shape[0]):
        x_train[trial, 
                x_train_spike_times[trial]["x"], 
                x_train_spike_times[trial]["t"]] = 1

    print("loading testing")
    for trial in trange(x_test.shape[0]):
        x_test[trial, 
               x_test_spike_times[trial]["x"], 
               x_test_spike_times[trial]["t"]] = 1

    print("loading validation")
    for trial in trange(x_validation.shape[0]):
        x_validation[trial, 
                     x_validation_spike_times[trial]["x"], 
                     x_validation_spike_times[trial]["t"]] = 1

    return x_train, y_train, x_test, y_test, x_validation, y_validation
