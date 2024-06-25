# temp gsc loader whilst awaiting file backup from perceptron

# import modules 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
def load_gsc(dataset_directory,
             NETWORK_SCALE,
             NUM_INPUT,
             num_samples,
             to_concatenate, 
             num_frames = 20):
    
    params = {}
    params["dataset_directory"] = dataset_directory
    params["NETWORK_SCALE"] = NETWORK_SCALE
    params["NUM_INPUT"] = NUM_INPUT

    x_train = np.load(os.path.expanduser(params.get("dataset_directory")) + "training_x_data.npy")
    y_train = np.load(os.path.expanduser(params.get("dataset_directory")) + "training_y_data.npy")
    
    x_test = np.load(os.path.expanduser(params.get("dataset_directory")) + "testing_x_data.npy")
    y_test = np.load(os.path.expanduser(params.get("dataset_directory")) + "testing_y_data.npy")
    
    #TODO: Redundant? any point to have a NUM_INPUT if data can be obtained through dataset shape
    assert x_train.shape[1] == params.get("NUM_INPUT"), "dataset input size doesn't match passed input parameter size"
    
    if params.get("NETWORK_SCALE") < 1:
        assert len(x_train) == len(y_train)
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        print(f"original network size: {len(x_train)}")
        x_train = x_train[:int(len(x_train) * params.get("NETWORK_SCALE"))]
        y_train = y_train[:int(len(y_train) * params.get("NETWORK_SCALE"))]
        print(f"reduced network size: {len(x_train)}")
        print("!! network reduced")

    training_images = np.swapaxes(x_train, 1, 2) 
    testing_images = np.swapaxes(x_test, 1, 2) 
 
    training_images = training_images + abs(np.floor(training_images.min()))
    testing_images = testing_images + abs(np.floor(testing_images.min()))

    training_labels = y_train
    testing_labels = y_test
    
    # adding validation data if exists
    validation_images = np.array([])
    validation_labels = np.array([])
    if os.path.isfile(os.path.expanduser(params.get("dataset_directory")) + "validation_y_data.npy"):
        print("!! validation dataset loaded successfully")
        x_validation = np.load(os.path.expanduser(params.get("dataset_directory")) + "validation_x_data.npy")
        y_validation = np.load(os.path.expanduser(params.get("dataset_directory")) + "validation_y_data.npy")
        
        validation_images = np.swapaxes(x_validation, 1, 2) 
        validation_images = validation_images + abs(np.floor(validation_images.min()))
        
        validation_labels = y_validation

    # crop the num of samples
    train_x, train_y = training_images[:num_samples], training_labels[:num_samples]
    validation_x, validation_y = validation_images[:num_samples], validation_labels[:num_samples]
    test_x, test_y = testing_images[:num_samples], testing_labels[:num_samples]

    # repeat frames for current injection over (frame) time
    train_x = np.repeat(train_x, num_frames, axis = 1)
    validation_x = np.repeat(validation_x, num_frames, axis = 1)
    test_x = np.repeat(test_x, num_frames, axis = 1)

    # shuffle array before stacking (stacking like ring buffer) 
    shuffler = np.random.permutation(len(train_x))
    train_x = train_x[shuffler]
    train_y = train_y[shuffler]
    shuffler = np.random.permutation(len(validation_x))
    validation_x = validation_x[shuffler]
    validation_y = validation_y[shuffler]
    shuffler = np.random.permutation(len(test_x))
    test_x = test_x[shuffler]
    test_y = test_y[shuffler]

    # to concatenate
    if to_concatenate:
        train_x = np.concatenate(train_x, axis = 0)
        validation_x = np.concatenate(validation_x, axis = 0)
        test_x = np.concatenate(test_x, axis = 0)

    return train_x, train_y, validation_x, validation_y, test_x, test_y

train_x, train_y, validation_x, validation_y, test_x, test_y = load_gsc("/its/home/ts468/data/rawSC/rawSC_80input/", 
                                                                        1, 
                                                                        80,
                                                                        10,
                                                                        False)

#plt.imshow(np.rot90(train_x), aspect="auto")
#plt.savefig("temp.png")

#print(train_x.shape)