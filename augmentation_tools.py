import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from tqdm import trange
import random
import imageio as iio
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops
import pandas as pd


def shift_y_axis(image_array, shift_value = np.random.randint(-5, 5)):
    """
    Shifts the passed array on the y axis

    :param image_array: image array to be shifted
    :param shift_value: value to be shifted by, if blank, will be shifted randomly between -5 and 5
    :return: new image array with shifted values
    """
    new_image_array = copy.deepcopy(image_array)
    if shift_value > 0:
        new_image_array[0: -shift_value, :] = image_array[shift_value:,:]
        new_image_array[-shift_value:,:] = 0
    if shift_value < 0:
        new_image_array[-shift_value:, :] = image_array[0: shift_value, :]
        new_image_array[0:-shift_value, :] = 0
    return new_image_array


def shift_x_axis(image_array, shift_value = np.random.randint(-5, 5)):
    """
    Shifts the passed array on the x axis

    :param image_array: image array to be shifted
    :param shift_value: value to be shifted by, if blank, will be shifted randomly between -5 and 5
    :return: new image array with shifted values
    """
    new_image_array = copy.deepcopy(image_array)
    if shift_value > 0:
        new_image_array[0: ,shift_value:] = image_array[:,: -shift_value]
        new_image_array[:, :shift_value] = 0
    if shift_value < 0:
        new_image_array[:,:shift_value] = image_array[:,-shift_value:]
        new_image_array[:,shift_value:] = 0
    return new_image_array


def neighbour_swap(image_array, pSwap = 0.2, kSwap = 3):
    """
    Randomly switches values with neighbouring values

    :param image_array: image array to have values switched
    :param spSwap: Probability of swapping, default is 0.2
    :param kSwap: Distance of neighbours to swap, default is 3
    :return: new image array with swapped values
    """
    new_image_array = copy.deepcopy(image_array)
    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            if np.random.uniform() > pSwap:
                k = np.random.randint(-kSwap, kSwap, 2)
                kx, ky = x + k[0], y + k[1]
                if kx > -1 and kx < image_array.shape[0] and ky > -1 and ky < image_array.shape[1]:
                    new_image_array[kx, ky] = image_array[x, y]
    return new_image_array


def get_image_center_of_mass(image_array):
    """
    Getting the centre of mass of a given image array

    :param image array: 1D Image array
    :return: centre of mass, weighted centre of mass
    """
    threshold_value = filters.threshold_otsu(image_array)
    labeled_foreground = (image_array > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image_array)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    return center_of_mass, weighted_center_of_mass


def combine_two_images(image_array_1, image_array_2, com_or_wcom = 0):
    """
    To combine two images by averaging them together

    :param image_array_1: first image
    :param image_array_2: second image
    :param com_or_wcom: centre of mass or weighted centre of mass
    :return: image array of combined images
    """
    com1 = np.round(get_image_center_of_mass(image_array_1)[com_or_wcom])
    com2 = np.round(get_image_center_of_mass(image_array_2)[com_or_wcom])

    difference = np.round(np.mean([com1, com2], axis = 0))

    new_image1 = shift_y_axis(shift_x_axis(image_array_1, int((difference - com1)[1])), int((difference - com1)[0]))
    new_image2 = shift_y_axis(shift_x_axis(image_array_2, int((difference - com2)[1])), int((difference - com2)[0]))
    
    combined_image = copy.deepcopy(new_image1)

    for y in range(new_image1.shape[0]):
        for x in range(new_image1.shape[1]):
            combined_image[y, x] = (combined_image[y, x] + new_image2[y, x] / 2)
    
    return combined_image


def combine_two_images_and_concatinate(training_images, training_labels):

    # sort x training images into y categories for blending
    categories = [[] for i in range(np.max(training_labels) + 1)]

    for i in range(training_labels.shape[0]):
        categories[training_labels[i]].append(i)
        
    combined_image_array = []
    combined_class_array = []

    for spoken_word in trange(len(categories)):

        random.shuffle(categories[spoken_word])

        while len(categories[spoken_word]) > 1:
            image1 = training_images[categories[spoken_word][0]]
            image2 = training_images[categories[spoken_word][1]]
            
            combined_image_array.append(combine_two_images(image1, image2))
            categories[spoken_word].pop()
            categories[spoken_word].pop()
            
            combined_class_array.append(spoken_word)
    
    combined_training_images = np.concatenate([training_images, 
                                               np.array(combined_image_array)])
    
    combined_training_labels = np.concatenate([training_labels,
                                               np.array(combined_class_array)])
    
    # Shuffle in unison
    shuffler = np.random.permutation(len(combined_training_images))
    combined_training_images_shuffled = combined_training_images[shuffler]
    combined_training_labels_shuffled = combined_training_labels[shuffler]
    
    return combined_training_images_shuffled, combined_training_labels_shuffled

def duplicate_and_mod_dataset(training_details, training_images):
    training_images_repeat = np.repeat(training_images, 2, axis = 0)
    training_details_repeat = pd.DataFrame(np.repeat(training_details.values, 2, axis = 0))

    for trial in trange(0, len(training_details_repeat), 2):
        training_images_repeat[trial] = neighbour_swap(training_images_repeat[trial])
        
    return training_details_repeat, training_images_repeat, np.array(training_details_repeat.loc[:, 5], dtype = "int8")
    