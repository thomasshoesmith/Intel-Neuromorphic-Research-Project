# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class GSCDataset(Dataset):
    """PilotNet dataset class that preserver temporal continuity. Returns
    images and ground truth values when the object is indexed.

    Parameters
    ----------
    path : str
        Path of the dataset folder. If the folder does not exists, the folder
        is created and the dataset is downloaded and extracted to the folder.
        Defaults to '../data'.
    sequence : int
        Length of temporal sequence to preserve. Default is 16.
    transform : lambda
        Transformation function to be applied to the input image.
        Defaults to None.
    train : bool
        Flag to indicate training or testing set. Defaults to True.
    visualize : bool
        If true, the train/test split is ignored and the temporal sequence of
        the data is preserved. Defaults to False.

    Examples
    --------

    >>> dataset = PilotNetDataset()
    >>> images, gts = dataeset[0]
    >>> num_samples = len(dataset)
    """
    def __init__(
        self, path='data', sequence=16,
        train=True, visualize=False, transform=None,
        extract=True, download=True,
    ):
        self.path = path + '/driving_dataset/'

        with open(self.path + 'data.txt', 'r') as data:
            all_samples = [line.split() for line in data]

        self.samples = all_samples

        if visualize is True:
            inds = np.arange(len(all_samples)//sequence)
        else:
            inds = np.random.RandomState(
                    seed=42
                ).permutation(len(all_samples)//sequence)
        if train is True:
            self.ind_map = inds[
                    :int(len(all_samples) / sequence * 0.8)
                ] * sequence
        else:
            self.ind_map = inds[
                    -int(len(all_samples) / sequence * 0.2):
                ] * sequence

        self.sequence = sequence
        self.transform = transform

    def __getitem__(self, index: int):
        images = []
        gts = []
        for i in range(self.sequence):
            path, gt = self.samples[self.ind_map[index] + i]

            image = Image.open(self.path + path)
            gt_val = float(gt) * np.pi / 180
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
            gts.append(torch.tensor(gt_val, dtype=image.dtype))

        images = torch.stack(images, dim=3)
        gts = torch.stack(gts, dim=0)

        return images, gts

    def __len__(self):
        return len(self.ind_map)
