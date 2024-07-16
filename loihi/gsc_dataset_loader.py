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
             scale_value = 0.0009, 
             num_frames = 10,
             shuffle = True):
    
    print("loading ...")
    params = {}
    params["dataset_directory"] = dataset_directory
    params["NETWORK_SCALE"] = NETWORK_SCALE
    params["NUM_INPUT"] = NUM_INPUT

    train_x = np.load(os.path.expanduser(params.get("dataset_directory")) + "training_x_data.npy")
    train_y = np.load(os.path.expanduser(params.get("dataset_directory")) + "training_y_data.npy")
    
    test_x = np.load(os.path.expanduser(params.get("dataset_directory")) + "testing_x_data.npy")
    test_y = np.load(os.path.expanduser(params.get("dataset_directory")) + "testing_y_data.npy")
    
    #TODO: Redundant? any point to have a NUM_INPUT if data can be obtained through dataset shape
    assert train_x.shape[1] == params.get("NUM_INPUT"), "dataset input size doesn't match passed input parameter size"
    if params.get("NETWORK_SCALE") < 1:
        assert len(train_x) == len(train_y)
        p = np.random.permutation(len(train_x))
        train_x, train_y = train_x[p], train_y[p]
        print(f"original network size: {len(train_x)}")
        train_x = train_x[:int(len(train_x) * params.get("NETWORK_SCALE"))]
        train_y = train_y[:int(len(train_y) * params.get("NETWORK_SCALE"))]
        print(f"reduced network size: {len(train_x)}")
        print("!! network reduced")

    # swap axes (maybe need to transpose)
    train_x = np.swapaxes(train_x, 1, 2) 
    test_x = np.swapaxes(test_x, 1, 2) 
 
    # move values into positive
    train_x = train_x + abs(np.floor(train_x.min()))
    test_x = test_x + abs(np.floor(test_x.min()))
    
    # adding validation data if exists
    validation_x = np.array([])
    validation_y = np.array([])
    if os.path.isfile(os.path.expanduser(params.get("dataset_directory")) + "validation_y_data.npy"):
        print("!! validation dataset loaded successfully")
        validation_x = np.load(os.path.expanduser(params.get("dataset_directory")) + "validation_x_data.npy")
        validation_y = np.load(os.path.expanduser(params.get("dataset_directory")) + "validation_y_data.npy")
        
        validation_x = np.swapaxes(validation_x, 1, 2) 
        validation_x = validation_x + abs(np.floor(validation_x.min()))

    if shuffle:
        # shuffle array before stacking (stacking like ring buffer) 
        shuffler = np.random.permutation(len(train_x))
        train_x = train_x[shuffler]
        train_y = train_y[shuffler]
        # shuffler = np.random.permutation(len(validation_x))
        # validation_x = validation_x[shuffler]
        # validation_y = validation_y[shuffler]
        # shuffler = np.random.permutation(len(test_x))
        # test_x = test_x[shuffler]
        # test_y = test_y[shuffler]

    # crop the num of samples
    train_x, train_y = train_x[:num_samples], train_y[:num_samples]
    validation_x, validation_y = validation_x[:num_samples], validation_y[:num_samples]
    test_x, test_y = test_x[:num_samples], test_y[:num_samples]

    # repeat frames for current injection over (frame) time
    train_x = np.repeat(train_x, num_frames, axis = 1)
    validation_x = np.repeat(validation_x, num_frames, axis = 1)
    test_x = np.repeat(test_x, num_frames, axis = 1)

    # scale values 
    train_x = train_x * scale_value
    validation_x = validation_x * scale_value
    test_x = test_x * scale_value

    # to concatenate
    if to_concatenate:
        train_x = np.concatenate(train_x, axis = 0)
        validation_x = np.concatenate(validation_x, axis = 0)
        test_x = np.concatenate(test_x, axis = 0)

    #plt.figure()
    #plt.imshow(train_x[2000:4000,:].T)
    #plt.gca().set_box_aspect(1.0)
    #plt.gca().set_aspect('auto')
    #plt.show()
    return train_x, train_y, validation_x, validation_y, test_x, test_y


"""
train_x, train_y, validation_x, validation_y, test_x, test_y = load_gsc("/its/home/ts468/data/rawSC/rawSC_80input/", 
                                                                        1, 
                                                                        80,
                                                                        10,
                                                                        False)
"""

#plt.imshow(np.rot90(train_x), aspect="auto")
#plt.savefig("temp.png")

#print(train_x.shape)
