import numpy as np
import os
import torch
from torch.utils.data import Dataset


def load_data_GSC(dataset_dir,
                  network_scale,
                  num_samples = None):
    
    """
    loads Google Speexch Commands dataset.
    looks for directory contraining the mel spectrogram

    :param dataset_dir: location of dataset
    :param network_scale: scale of the dataset returned (0 - 1)
    :param num_samples: how many samples to return
    :return: (x_train, y_train, x_test, y_test)
    """ 

    dir = dataset_dir #"/mnt/data0/ts468/data/rawSC/rawSC_80input/"
    x_train = np.load(os.path.expanduser(dir) + "training_x_data.npy")
    y_train = np.load(os.path.expanduser(dir) + "training_y_data.npy")
    x_test = np.load(os.path.expanduser(dir) + "testing_x_data.npy")
    y_test = np.load(os.path.expanduser(dir) + "testing_y_data.npy")

    # used to scale down the dataset for contraint runs
    if network_scale < 1:
            assert len(x_train) == len(y_train)
            p = np.random.permutation(len(x_train))
            x_train, y_train = x_train[p], y_train[p]
            print(f"original network size: {len(x_train)}")
            x_train = x_train[:int(len(x_train) * network_scale)]
            y_train = y_train[:int(len(y_train) * network_scale)]
            print(f"reduced network size: {len(x_train)}")
            print("!! network reduced")

    # to invert the dataset (emphasised with positive values, not negative)
    x_train = x_train + abs(np.floor(x_train.min()))
    x_test = x_test + abs(np.floor(x_test.min()))

    # scaled to the number of samples passed
    return (x_train[:num_samples], y_train[:num_samples], 
            x_test[:num_samples], y_test[:num_samples])


class GSCDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    

def rescale_factor(w, bits):
    rng = float(2**(bits-1))
    mx = max(np.percentile(w,99), np.percentile(-w,99))
    fac = (rng-1)/mx
    mn = -(rng-2)/fac
    return (fac, mn, mx)
