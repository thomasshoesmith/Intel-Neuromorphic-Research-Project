# temp gsc loader whilst awaiting file backup from perceptron

# import modules 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
def load_gsc(dataset_directory,
             num_samples = None,
             to_concatenate = False, 
             num_frames = 1,
             shuffle = True):

    train_x = np.load(os.path.expanduser(dataset_directory) + "training_x_data.npy")
    train_y = np.load(os.path.expanduser(dataset_directory) + "training_y_data.npy")
    
    test_x = np.load(os.path.expanduser(dataset_directory) + "testing_x_data.npy")
    test_y = np.load(os.path.expanduser(dataset_directory) + "testing_y_data.npy")
   
    # adding validation data if exists
    validation_x = np.array([])
    validation_y = np.array([])
    if os.path.isfile(os.path.expanduser(dataset_directory) + "validation_y_data.npy"):
        print("!! validation dataset loaded successfully !!")
        validation_x = np.load(os.path.expanduser(dataset_directory) + "validation_x_data.npy")
        validation_y = np.load(os.path.expanduser(dataset_directory) + "validation_y_data.npy")

    # shuffle array before stacking (stacking like ring buffer) 
    if shuffle:
        shuffler = np.random.permutation(len(train_x))
        train_x = train_x[shuffler]
        train_y = train_y[shuffler]

        # unnecessary, but being used for debugging 
        shuffler = np.random.permutation(len(train_x))
        train_x = train_x[shuffler]
        train_y = train_y[shuffler]

    # crop the num of samples
    train_x, train_y = train_x[:num_samples], train_y[:num_samples]
    validation_x, validation_y = validation_x[:num_samples], validation_y[:num_samples]
    test_x, test_y = test_x[:num_samples], test_y[:num_samples]

    # repeat frames for current injection over (frame) time
    train_x = np.repeat(train_x, num_frames, axis = 1)
    validation_x = np.repeat(validation_x, num_frames, axis = 1)
    test_x = np.repeat(test_x, num_frames, axis = 1)

    # to concatenate
    if to_concatenate:
        train_x = np.concatenate(train_x, axis = 0)
        validation_x = np.concatenate(validation_x, axis = 0)
        test_x = np.concatenate(test_x, axis = 0)

    return train_x, train_y, validation_x, validation_y, test_x, test_y