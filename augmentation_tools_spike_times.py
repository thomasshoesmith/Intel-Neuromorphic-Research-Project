import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import re
import copy

def augmentation_y_shift(array_x, 
                         array_y, 
                         shift_value = np.random.randint(-5, 5), 
                         x_lim = 1000,
                         percentage_added = 0.5, 
                         percentage_skipped = 0.2):
    """_summary_

    Args:
        array_x (_type_): _description_
        array_y (_type_): _description_
        shift_value (_type_, optional): _description_. Defaults to np.random.randint(-5, 5).
        percentage_added (float, optional): How much of array_x gets shifted. Defaults to 0.5.
        percentage_skipped (float, optional): How much of array_x does not get augmented. Defaults to 0.5.
    """
    
    #focusing on skipped only.
    new_image_array = copy.deepcopy(array_x)
    for trial in new_image_array:
        if np.random.rand() < percentage_skipped: continue
        trial["t"] += np.random.randint(-5, 5)
        trial = np.delete(trial, 
                            list(np.where(trial["t"] > x_lim)[0]) + \
                            list(np.where(trial["t"] < 0)[0]))
            
    return array_x, array_y
        
def merge_and_return_a_new(array_x,
                           array_y,
                           x_lim = 1600,
                           y_lim = 80,
                           percentage_added = 0.5):
    """Augmentation for combining two images and returning a 'merged' combined dataset

    Args:
        array_x (_type_): spike array (formatted as a [[(x, t, p)...spike times...]...array....])
        array_y (_type_): y label for array x
        x_lim (int, optional): x limit for neuron range within layer. Defaults to 1600.
        y_lim (int, optional): y limit for spike time range within trial. Defaults to 80.
        percentage_added (float, optional): how many new merged images are appended to dataset, 50% = 50% increase when returned. Defaults to 0.5.
    """
    
    new_array_x = copy.deepcopy(array_x)
    new_array_y = copy.deepcopy(array_y)
    
    appended_array_x = []
    appended_array_y = []
    
    # loop through all categories
    for category in np.unique(new_array_y):
        
        # percentage added (0.5 adds 50%, 1.0 adds 100% ...)
        for i in range(int(np.where(new_array_y == category)[0].shape[0] * percentage_added)):
            
            # checking unique images to merge
            image_1_id, image_2_id = 0, 0
            while image_1_id == image_2_id:
                image_1_id = np.where(new_array_y == category)[0][np.random.randint(0, np.where(new_array_y == category)[0].shape[0])]
                image_2_id = np.where(new_array_y == category)[0][np.random.randint(0, np.where(new_array_y == category)[0].shape[0])]
                
            # shifting along x axis by half the difference for each image    
            new_array_x[image_1_id]["t"] -= int((np.sum(new_array_x[image_1_id]["t"]) / 
                                                 new_array_x[image_1_id]["t"].shape[0] - np.sum(new_array_x[image_2_id]["t"]) / 
                                                 new_array_x[image_2_id]["t"].shape[0]) / 2)
            
            new_array_x[image_2_id]["t"] += int((np.sum(new_array_x[image_1_id]["t"]) / 
                                                 new_array_x[image_1_id]["t"].shape[0] - np.sum(new_array_x[image_2_id]["t"]) / 
                                                 new_array_x[image_2_id]["t"].shape[0]) / 2)

            # deleting out of bounds shifted times
            new_array_x[image_1_id] = np.delete(new_array_x[image_1_id], 
                                list(np.where(new_array_x[image_1_id]["t"] > x_lim)[0]) + \
                                list(np.where(new_array_x[image_1_id]["t"] < 0)[0]))
            new_array_x[image_2_id] = np.delete(new_array_x[image_2_id], 
                                list(np.where(new_array_x[image_2_id]["t"] > x_lim)[0]) + \
                                list(np.where(new_array_x[image_2_id]["t"] < 0)[0]))
            
            # merge both images by skipping every other spike time, followed by sorting by spike time
            x = np.sort(np.concatenate((new_array_x[image_1_id][1::2], new_array_x[image_2_id][1::2])), order = "t")
            
            appended_array_x.append(x)
            appended_array_y.append(category)
        
    appended_array_x = np.array(appended_array_x, dtype = 'object')
    
    return (np.concatenate((appended_array_x, new_array_x)), 
            np.concatenate((appended_array_y, new_array_y)))