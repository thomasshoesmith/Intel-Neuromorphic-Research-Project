import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from rawHD_dataset_loader_padded_spikes import rawHD_Loader
from scipy.signal import convolve2d
from tqdm import trange
from copy import deepcopy

params = {} 
params["dataset_directory"] = "/raw-spiking-heidleberg-digits-80input/"
params["num_samples"] = None
params["verbose"] = False

x_train = np.load("/its/home/ts468/PhD/Intel-Neuromorphic-Research-Project/raw-spiking-heidleberg-digits-80input/training_x_spikes.npy", allow_pickle = True)
y_train = np.load("/its/home/ts468/PhD/Intel-Neuromorphic-Research-Project/raw-spiking-heidleberg-digits-80input/training_y_spikes.npy", allow_pickle = True)
training_details = pd.read_csv(os.getcwd() + params.get("dataset_directory") + "training_details.csv")

speakers_list = np.array(list(training_details.loc[:, "Speaker"]))

def exponential_kernel_2d(size_t, size_x, tau_t, tau_x):
    t = np.linspace(-size_t/2, size_t/2, size_t + 1, dtype='int8')
    x = np.linspace(-size_x/2, size_x/2, size_x + 1, dtype = 'int8')
    
    T, X = np.meshgrid(t, x, indexing='ij')
    
    kernel = np.exp(-np.abs(T) / tau_t) * np.exp(-np.abs(X) / tau_x)
    
    # 3D Surface plot
    fig = plt.figure(figsize=(12, 5))
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(T, X, kernel, cmap='viridis', edgecolor='none')
    ax2.set_title("3D Surface Plot of Kernel")

    plt.show() 
    
    return kernel / np.sum(kernel)  # Normalize kernel

def gaussian_kernel_2d(size_t, size_x, sigma_t = 10, sigma_x = 10, display = False):

    t = np.arange(-size_t // 2, size_t // 2 + 1)
    x = np.arange(-size_x // 2, size_x // 2 + 1)

    T, X = np.meshgrid(t, x, indexing='ij')

    kernel = np.exp(-((T**2) / (2 * sigma_t**2) + (X**2) / (2 * sigma_x**2)))
    kernel /= np.sum(kernel)

    if display:
        # Plot kernel
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, X, kernel, cmap='viridis', edgecolor='none')
        ax.set_title("3D Surface Plot of Gaussian Kernel")
        
        plt.show()
    
    return kernel

def get_vr_distance_2d(spike_train1, 
                       spike_train2, 
                       tau_t = 40, 
                       tau_x = 5,
                       size_t = 40,
                       size_x = 10, 
                       last_spike_t = 1700, 
                       num_neurons = 80,
                       display = False,
                       second_kernel_scale = 1.0):
    
    time_bins = np.arange(0, last_spike_t + 1, 1)  # 1 ms bins
    spike_matrix1 = np.zeros((len(time_bins), num_neurons))
    spike_matrix2 = np.zeros((len(time_bins), num_neurons))

    for t, x in zip(spike_train1['t'], spike_train1['x']):
        spike_matrix1[int(t), int(x)] = 1
    for t, x in zip(spike_train2['t'], spike_train2['x']):
        spike_matrix2[int(t), int(x)] = 1
    
    #kernel = exponential_kernel_2d(size_t = size_t, size_x = size_x, tau_t=tau_t, tau_x=tau_x)
    kernel = gaussian_kernel_2d(size_t = size_t, size_x = size_x, sigma_x = tau_x, sigma_t = tau_t, display = display)
    
    smoothed1 = convolve2d(spike_matrix1, kernel, mode='same', boundary='wrap')
    smoothed2 = convolve2d(spike_matrix2, kernel * second_kernel_scale, mode='same', boundary='wrap')
    
    # Compute Euclidean distance between smoothed spike matrices
    distance = np.linalg.norm(smoothed1 - smoothed2)
    
    return distance, smoothed1, kernel

# iterate through all digits of speaker x, and get COM

def normalise_dataset(x_train, 
                      y_train,
                      x_lim = 80,
                      t_lim = 1600):
    
    x_train_new = deepcopy(x_train)
    
    for speaker in np.unique(speakers_list): 
        for digit in np.unique(y_train):

            # get COM of each digit spoken by speaker
            t_com_across_speaker_digit, x_com_across_speaker_digit = [], []

            for index in range(y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]):

                t_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]["t"])
                x_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]["x"])

                t_com_across_speaker_digit.append(t_com)
                x_com_across_speaker_digit.append(x_com)
                
            #print(f" mean COM for t : {int(np.mean(t_com_across_speaker_digit))}")
            #print(f" mean COM for x : {int(np.mean(x_com_across_speaker_digit))}")
            
            # shift on both x and t
            for index in range(y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]):
                x_train_array = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]
                
                x_train_array["t"] += int(np.mean(t_com_across_speaker_digit)) - int(t_com_across_speaker_digit[index])
                x_train_array["t"] += int(np.mean(x_com_across_speaker_digit)) - int(x_com_across_speaker_digit[index])
                
                #x_train_array = x_train_array[x_train_array["x"] >= 0]
                #x_train_array = x_train_array[x_train_array["x"] < x_lim]
                x_train_array = x_train_array[x_train_array["t"] >= 0]
                x_train_array = x_train_array[x_train_array["t"] < t_lim]
                
                x_train_new[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index] = deepcopy(x_train_array)
                
    return x_train_new, y_train

# normalise all data
x, y = normalise_dataset(x_train, y_train)

intra_VR_distance_mean = np.zeros((10, 20))
intra_VR_distance_std = np.zeros((10, 20))
count = 0

for speaker_index, speaker in enumerate(np.unique(speakers_list)):
    for digit in np.unique(y_train):
        distance_across_digits_from_speaker = []
        for index_to_compare in trange(y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]):
            for index in range(y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]):
                distance, S, _ = get_vr_distance_2d(x[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index],
                                                    x[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_to_compare],
                                                    display = False,
                                                    size_t = 80)
                
                distance_across_digits_from_speaker.append(distance)

        intra_VR_distance_mean[speaker_index, digit] = np.mean(distance_across_digits_from_speaker)
        intra_VR_distance_std[speaker_index, digit] = np.std(distance_across_digits_from_speaker)
        
        print(f"comparison {count} of 199")
        count += 1
        np.save("rawHD_analyse_training_data_intra_VR_distance_mean.npy", intra_VR_distance_mean)
        np.save("rawHD_analyse_training_data_intra_VR_distance_std.npy", intra_VR_distance_std)