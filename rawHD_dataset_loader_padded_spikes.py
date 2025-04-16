###############################################
#               Dataset loader                #
#       to a numpy array of fixed memory      #
###############################################

import numpy as np
import os
from tqdm import trange
import pandas as pd

def rawHD_Loader(dir, 
                 num_samples = None,
                 return_single_sample = None,
                 shuffle = False,
                 shuffle_seed = None,
                 number_of_neurons = 80,
                 number_of_timesteps = 2000,
                 process_padded_spikes = True,
                 validation_split = 0.2):
    
    # loading x spike times
    x_train = np.load(os.path.expanduser(dir) + "training_x_spikes.npy", 
                      allow_pickle = True)
    x_test = np.load(os.path.expanduser(dir) + "testing_x_spikes.npy", 
                     allow_pickle = True)
    
    # loading y labels to returning
    y_train = np.load(os.path.expanduser(dir) + "training_y_spikes.npy", 
                      allow_pickle = True)
    y_test = np.load(os.path.expanduser(dir) + "testing_y_spikes.npy", 
                     allow_pickle = True)
    
    # loading z labels (speakers)
    training_details = pd.read_csv(os.path.expanduser(dir) + "training_details.csv")
    testing_details = pd.read_csv(os.path.expanduser(dir) + "testing_details.csv")
    z_train = np.array(list(training_details.loc[:, "Speaker"]))
    z_test = np.array(list(testing_details.loc[:, "Speaker"]))
    
    # validation split
    x_validation = x_train[:int(validation_split * x_train.shape[0])]
    y_validation = y_train[:int(validation_split * y_train.shape[0])]
    z_validation = z_train[:int(validation_split * z_train.shape[0])]
    x_train = x_train[int(validation_split * x_train.shape[0]):]
    y_train = y_train[int(validation_split * y_train.shape[0]):]
    z_train = z_train[int(validation_split * z_train.shape[0]):]
    
    # shuffle array
    if shuffle:
        np.random.seed(shuffle_seed)
        shuffler = np.random.permutation(len(x_train))
        x_train = x_train[shuffler]
        y_train = y_train[shuffler]
        z_train = z_train[shuffler]
        
        # unnecessary, but being used for debugging 
        np.random.seed(shuffle_seed)
        shuffler = np.random.permutation(len(x_test))
        x_test = x_test[shuffler]
        y_test = y_test[shuffler]
        z_test = z_test[shuffler]
        
        

    # crop the num of samples
    if return_single_sample is None:
        x_train, y_train = x_train[:num_samples], y_train[:num_samples]
        x_validation, y_validation = x_validation[:num_samples], y_validation[:num_samples]
        x_test, y_test = x_test[:num_samples], y_test[:num_samples]
        
    # return specific value for testing / viewing
    # ugly by saves reshaping?
    if return_single_sample is not None:
        x_train, y_train = x_train[return_single_sample: return_single_sample + 1], \
                           y_train[return_single_sample: return_single_sample + 1]
        x_validation, y_validation = x_validation[return_single_sample: return_single_sample + 1], \
                                     y_validation[return_single_sample: return_single_sample + 1]
        x_test, y_test = x_test[return_single_sample: return_single_sample + 1], \
                         y_test[return_single_sample: return_single_sample + 1]
                         
        z_test, z_validation = z_test[return_single_sample: return_single_sample + 1], \
                               z_validation[return_single_sample: return_single_sample + 1]

    # define the empty zeroed arrays
    x_train_new = np.zeros((x_train.shape[0], 
                            number_of_neurons, 
                            number_of_timesteps), 
                           dtype = np.int8)
    x_test_new = np.zeros((x_test.shape[0], 
                           number_of_neurons, 
                           number_of_timesteps), 
                          dtype = np.int8)
    x_validation_new = np.zeros((x_validation.shape[0], 
                                 number_of_neurons, 
                                 number_of_timesteps), 
                                dtype = np.int8)

    if not process_padded_spikes:
        return x_train, y_train, z_train, x_test, y_test, z_test, x_validation, y_validation, z_validation

    # flipping bits
    print("loading training")
    for trial in trange(x_train.shape[0]):
        x_train_new[trial, 
                    x_train[trial]["x"], 
                    x_train[trial]["t"]] = 1

    print("loading testing")
    for trial in trange(x_test.shape[0]):
        x_test_new[trial, 
                   x_test[trial]["x"], 
                   x_test[trial]["t"]] = 1

    print("loading validation")
    for trial in trange(x_validation.shape[0]):
        x_validation_new[trial, 
                         x_validation[trial]["x"], 
                         x_validation[trial]["t"]] = 1

    return x_train_new, y_train, z_train, x_test_new, y_test, z_test, x_validation_new, y_validation, z_validation