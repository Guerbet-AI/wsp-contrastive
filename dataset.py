
# coding=utf-8
import random

import numpy
from torch.utils.data import Dataset
import torchvision
import numpy as np
from os.path import join, isdir, isfile
import os as os
import pandas as pd
import psutil
import torch
import nibabel as nib
import math
import itertools
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt

class DatasetCL(Dataset):  # output size : (N,(512,512,2),1,1)

    def __init__(self, config, training: bool = False, validation: bool = False, *args, **kwargs):

        """PyTorch Dataset Module for training and validation
        Keywords arguments:
        :param config: a Config file that must contain the required arguments (see config.py)
        :param training: boolean if training phase
        :param validation: boolean if validation phase
        :return: a PyTorch Dataset
        """

        super().__init__(*args, **kwargs)
        assert training != validation

        self.config = config

        self.df = pd.read_csv(join(self.config.lght_dir, 'dataframe.csv'), delimiter=",",
                                      index_col='subject').drop(['Unnamed: 0'], axis=1)
        dct = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        self.df['label'] = [*map(dct.get, self.df['class'])]

        if training:
            self.labels = self.df[self.df[config.train_set] == 1]
            self.path = join(self.config.path_to_data, 'train')
        elif validation:
            self.labels = self.df[self.df[config.val_set] == 1]
            self.path = join(self.config.path_to_data, 'validation')

        self.subject_ids = self.labels.index                    # subject IDs in the dataset
        self.imgs_loaded = [self.load_img(k) for k in self.subject_ids]
                                                                # we load each volume
        self.slices = np.dstack(self.imgs_loaded)               # shape (512,512,n_slices)
        self.n_slices = self.slices.shape[-1]
        vols = [self.imgs_loaded[k].shape[-1] * [self.subject_ids[k]] for k in range(len(self.imgs_loaded))]
                                                                # we repeat the names of the subject IDs,
                                                                # according to the number of slices.
        self.volumes = list(itertools.chain(*vols))             # shape (n_slices): list of volume names repeated
                                                                # each volume length.
        u = dict(Counter(self.volumes))
        self.z_pos = list(itertools.chain(*[[x / i for x in list(range(0, i))] for i in u.values()]))
        # d parameter for WSP loss. Corresponds to the normalized 0-1 float of the slice position in the volume.

        if validation:
            # if validation, we shuffle the dataset once
            # otherwise for each batch you will have slices from the same patient
            # so exp(z_i,z_j) / sum_k(exp(z_i,z_k)) will be very low, so high loss value
            l = list(range(self.n_slices))
            self.shuffled_idx = random.sample(l, len(l))
            self.slices = self.slices[:, :, self.shuffled_idx]
            self.volumes = list(itemgetter(*self.shuffled_idx)(self.volumes))
            self.z_pos = list(itemgetter(*self.shuffled_idx)(self.z_pos))

        self.classes = [int(self.labels.loc[x, 'class']) for x in np.unique(self.volumes)]
        self.class_sample_count = np.array(
            [len(np.where(self.classes == t)[0]) for t in np.unique(self.classes)])
        self.class_sample_count = dict(Counter(self.classes))

        self.weight = {k: 1/v for k, v in self.class_sample_count.items()}
        self.samples_weight = np.array([self.weight[t] for t in self.classes])
        self.samples_weight = torch.from_numpy(self.samples_weight)


        self.indices = {}
        for z in self.volumes:
            self.indices[z] = [i for i, x in enumerate(self.volumes) if x == z]

        print(self.n_slices)
        print('Class repartition',self.class_sample_count)


    def load_img(self, k: str = None, pct: float = 0.7):

        """Function to load an image given a Nifty file and a percentage of volume to keep around the center
        Keywords arguments:
        :k: path to a Nifty file
        :pct: float between 0 and 1 representing the percentage around the center of the volume to keep
        :return: a NumPy array with the selected slices"""

        mat = np.asanyarray(nib.load(join(self.path, k)).dataobj)
        length = mat.shape[-1]
        u = range(int((length * (1-pct)) // 2), int(length * pct + int((length * (1-pct)) // 2 ) ) )
        return mat[:,:,u]


    def collate_fn(self, list_samples: list = None):

        list_x = torch.stack(
            [torch.as_tensor(x.astype('uint8').copy(), dtype=torch.float) for (x, y, z, m) in list_samples], dim=0)     # dimension finale: (batch_size, 1, 512, 512)
        if self.config.pretrained:
            list_x = torch.repeat_interleave(list_x, 3, dim=1)
        list_y = torch.stack([torch.as_tensor(int(y), dtype=torch.long) for (x, y, z, m) in list_samples], dim=0)
        list_m = torch.stack([torch.as_tensor(m) for (x, y, z, m) in list_samples], dim=0)
        list_z = []
        for (x, y, z, m) in list_samples:
            list_z.append(z)

        return list_x, list_y, list_z, list_m


    def __getitem__(self, idx: int = None):

        data = self.slices[:, :, idx]       # shape(H,W)
        data = np.squeeze(data)[np.newaxis] # shape (1,H,W)
        subject_id = self.volumes[idx]
        label = self.labels.loc[subject_id, self.config.label_name]
        z = self.z_pos[idx]

        return data, label, subject_id, z   # output: the two augmented images, labels, ID

    def __len__(self):
        return self.slices.shape[-1]