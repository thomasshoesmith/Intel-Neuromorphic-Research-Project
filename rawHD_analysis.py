# %% [markdown]
# todo: add boundaries for x and t ranges. 0-80, 0-1600

# %%
# load libraries
import tonic
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import trange

# %%
params = {} 
params["dataset_directory"] = "/raw-spiking-heidleberg-digits-80input/"
params["verbose"] = False

x_train = np.load("/its/home/ts468/PhD/Intel-Neuromorphic-Research-Project/raw-spiking-heidleberg-digits-80input/training_x_spikes.npy", allow_pickle = True)
y_train = np.load("/its/home/ts468/PhD/Intel-Neuromorphic-Research-Project/raw-spiking-heidleberg-digits-80input/training_y_spikes.npy", allow_pickle = True)
training_details = pd.read_csv(os.getcwd() + params.get("dataset_directory") + "training_details.csv")

speakers_list = np.array(list(training_details.loc[:, "Speaker"]))

# %%
# get a single sample
print(f"y data      : {y_train[0]}")
print(f"x data      : {x_train[0]}")
print(f"speaker     :{speakers_list[0]}")

# %%
speaker = 0
digit = 2

print(f"digit: {y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)]}")
print(f"speaker: {speakers_list[np.where(speakers_list == speaker)]}")

# %%
# average digit of each speaker
# speakers (11) with a sub array of digits (20)

average_speaker_digit = [[] for _ in range(max(speakers_list) + 1)]

for speaker in average_speaker_digit:
    for digit in np.unique(y_train):
        speaker.append([digit])
    
# %%
# combined digit x for speaker y

speaker = 1
digit = 5

x, t = [], []
combined_images = []

for index in range(y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]): 
    # iterating through all digit examples for specified speaker
    t += list(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]["t"])
    x += list(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]["x"])

# Create the structured array
spike_data = np.zeros(len(t), dtype=[('x', 'i1'), ('t', '<i2'), ('p', 'i1')])

# Populate the structured array
spike_data['t'] = np.array(t)[np.argsort(t)]
spike_data['x'] = np.array(x)[np.argsort(t)]
spike_data['p'] = 1

# %%
index_0 = 0
index_1 = 1

t_1 = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["t"]
x_1 = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["x"]

t_1_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["t"])
x_1_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["x"])

t_2 = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["t"]
x_2 = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["x"]

t_2_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["t"])
x_2_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["x"])

print(f" time_difference: {int(t_1_com - t_2_com)}, neuron_difference:  {int(x_1_com - x_2_com)}")

# %%
index_0 = 0
index_1 = 1

speaker = 1
digit = 5

t_1 = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["t"]
x_1 = x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["x"]

t_1_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["t"])
x_1_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_0]["x"])


x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["t"] += int(t_1_com - t_2_com)

t_2_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["t"])
x_2_com = np.mean(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index_1]["x"])

print(f" time_difference: {int(t_1_com - t_2_com)}, neuron_difference:  {int(x_1_com - x_2_com)}")




# %%
# iterate through all digits of speaker x, and get COM

def get_com_for_speaker_digit(speaker, digit, 
                              x_train = x_train, 
                              y_train = y_train,
                              x_lim = 80,
                              t_lim = 1600):

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
        
        x_train_array = x_train_array[x_train_array["x"] >= 0]
        x_train_array = x_train_array[x_train_array["x"] < x_lim]
        x_train_array = x_train_array[x_train_array["t"] >= 0]
        x_train_array = x_train_array[x_train_array["t"] < t_lim]

get_com_for_speaker_digit(speaker = 1,
                          digit = 5)


# %%
# normalise all data

for speaker in np.unique(speakers_list):
    for digit in np.unique(y_train):
        get_com_for_speaker_digit(speaker,
                                  digit)

# %%
# get average digits

average_speaker_digit = [[] for _ in range(max(speakers_list) + 1)]
average_count_speaker_digit = [[] for _ in range(max(speakers_list) + 1)]

for speaker in np.unique(speakers_list):
    average_digit = [[] for _ in range(len(np.unique(y_train)))]
    average_count_digit = [[] for _ in range(len(np.unique(y_train)))]
    
    for digit in np.unique(y_train):
        x, t = [], []

        for index in range(y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]): 
            # iterating through all digit examples for specified speaker
            t += list(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]["t"])
            x += list(x_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)][index]["x"])

        # Create the structured array
        spike_data = np.zeros(len(t), dtype=[('x', 'i1'), ('t', '<i2'), ('p', 'i1')])

        # Populate the structured array
        spike_data['t'] = np.array(t)[np.argsort(t)]
        spike_data['x'] = np.array(x)[np.argsort(t)]
        spike_data['p'] = 1
        
        average_digit[digit] = spike_data
        average_count_digit[digit] = y_train[np.where(speakers_list == speaker)][np.where(y_train[np.where(speakers_list == speaker)] == digit)].shape[0]
    
    average_speaker_digit[speaker] = average_digit
    average_count_speaker_digit[speaker] = average_count_digit
    

# %%
def exponential_kernel_2d(size_t, size_x, tau_t, tau_x):
    t = np.linspace(-size_t/2, size_t/2, size_t + 1, dtype='int8')
    x = np.linspace(-size_x/2, size_x/2, size_x + 1, dtype = 'int8')
    
    T, X = np.meshgrid(t, x, indexing='ij')
    
    kernel = np.exp(-np.abs(T) / tau_t) * np.exp(-np.abs(X) / tau_x)
    
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
                       last_spike_t = 2000, 
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


# %%
vr_distance_speaker_digit = np.zeros((20 * 10, 20 * 10))

unique_speaker_list = np.unique(speakers_list)
unique_digit_list = np.unique(y_train)

for speaker_1 in trange(len(unique_speaker_list)):
    for speaker_2 in range(len(unique_speaker_list)):
        for digit_1 in range(len(unique_digit_list)):
            for digit_2 in range(len(unique_digit_list)):
                print((speaker_1 * 20) + digit_1, (speaker_2 * 20) + digit_2)
                
                second_kernel_scale = average_count_speaker_digit[unique_speaker_list[speaker_1]][digit_1] / average_count_speaker_digit[unique_speaker_list[speaker_2]][digit_2]
                
                distance, S, _ = get_vr_distance_2d(average_speaker_digit[unique_speaker_list[speaker_1]][digit_1],
                                                    average_speaker_digit[unique_speaker_list[speaker_2]][digit_2],
                                                    display = False,
                                                    size_t = 80)

                vr_distance_speaker_digit[(speaker_1 * 20) + digit_1, (speaker_2 * 20) + digit_2] = distance
        # save 'checkpoint' 
        np.save("vr_distance_speaker_digit.npy", vr_distance_speaker_digit)
        print("file saved")

# %%
np.save("vr_distance_speaker_digit.npy", vr_distance_speaker_digit)


