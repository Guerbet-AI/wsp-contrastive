import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data.sampler import Sampler
import itertools
from tqdm.auto import tqdm


class CustomSampler(Sampler):

    """A CustomSampler which aim is to provide a batch of volumes with no identical patient"""

    def __init__(self, dataset, batch_size, indices, k=10000, weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = k // batch_size
        self.subject_ids = np.unique(dataset.volumes)
        self.weights = weights
        self.indices = indices
        self.depth = 1

    def __iter__(self):

        self.batches = []

        for _ in tqdm(range(self.n_batches)):

            batch_idxs = []
            vols = []

            while len(batch_idxs) < self.batch_size:
                # sample in the volumes
                if self.weights is not None:
                    j = np.random.choice(range(len(self.subject_ids)),
                                         p=self.weights / np.sum(self.weights),
                                         size=1)[0]
                else:
                    j = random.choice(range(len(self.subject_ids)))
                vol = self.subject_ids[j]
                if vol not in vols:
                    # sample in the slices attributed to each volume
                    i = random.choice(self.indices[vol])
                    batch_idxs.append(i)
                    vols.append(vol)

            self.batches.append(batch_idxs)

        return iter(list(itertools.chain(*self.batches)))

    def __len__(self):
        return(len(self.n_batches))




class CustomSamplerTrain(Sampler):

    """A CustomSampler which aim is to provide a batch of volumes with no identical patient"""

    def __init__(self, dataset, batch_size, indices, k=10000, weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = k // batch_size
        self.subject_ids = np.unique(dataset.volumes)
        self.weights = weights
        self.indices = indices
        self.depth = 1

    def __iter__(self):

        self.batches = []

        for _ in tqdm(range(self.n_batches)):

            batch_idxs = []
            vols = []

            while len(batch_idxs) < self.batch_size:
                # sample in the volumes
                if self.weights is not None:
                    j = np.random.choice(range(len(self.subject_ids)),
                                         p=self.weights / np.sum(self.weights),
                                         size=1)[0]
                else:
                    j = random.choice(range(len(self.subject_ids)))
                vol = self.subject_ids[j]
                if vol not in vols:
                    i = random.choice(self.indices[vol])
                    """if i > np.min(self.indices[vol]) + 1 and i < np.max(self.indices[vol]) - 1:
                        range_i = range(max(i - 2, np.min(self.indices[vol])), min(i + 2, np.max(self.indices[vol])))"""
                    """
                    if i > np.min(self.indices[vol]) + 16 and i < np.max(self.indices[vol]) - 16:
                        range_i = range(i - 16, i + 16)
                    elif i == np.min(self.indices[vol]):
                        range_i = range(i, i + 32)
                    elif i == np.min(self.indices[vol]) + 1:
                        range_i = range(i - 1, i + 31)
                    elif i == np.max(self.indices[vol]):
                        range_i = range(i - 32, i)
                    elif i == np.max(self.indices[vol]) - 1:
                        range_i = range(i - 31, i + 1)"""

                    if i > np.min(self.indices[vol]) + 8 and i < np.max(self.indices[vol]) - 8:
                        range_i = range(i - 8, i + 8)
                    elif i >= np.min(self.indices[vol]) and i <= np.min(self.indices[vol]) + 8:
                        range_i = range(np.min(self.indices[vol]), np.min(self.indices[vol]) + 16)
                    elif i >= np.max(self.indices[vol]) - 16 and i <= np.max(self.indices[vol]):
                        range_i = range(np.max(self.indices[vol]) - 16, np.max(self.indices[vol]))

                    batch_idxs.append(range_i)
                    vols.append(vol)

            self.batches.append(batch_idxs)
            output = [item for sublist in self.batches for item in sublist]

        return iter(output)


    def __len__(self):
        return(len(self.n_batches))



class CustomSamplerVal(Sampler):

    """A CustomSampler which aim is to provide a batch of volumes with no identical patient"""

    def __init__(self, dataset, batch_size, indices, k=10000, weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = k
        self.subject_ids = np.unique(dataset.volumes)
        self.weights = weights
        self.indices = indices

    def __iter__(self):

        self.batches = []
        self.vols = []

        for i in tqdm(range(len(self.dataset))):

            """# sample in the volumes
            j = random.choice(range(len(self.subject_ids)))
            vol = self.subject_ids[j]
            # sample in the slices attributed to each volume
            i = random.choice(self.indices[vol])
            if i > np.min(self.indices[vol]) + 1 and i < np.max(self.indices[vol]) - 1:
                range_i = range(max(i-2,np.min(self.indices[vol])),min(i+2,np.max(self.indices[vol])))
                self.batches.append(range_i)
                self.vols.append(vol)"""
            vol = self.dataset.volumes[i]

            if i > np.min(self.indices[vol]) + 8 and i < np.max(self.indices[vol]) - 8:
                range_i = range(i - 8, i + 8)
            elif i >= np.min(self.indices[vol]) and i <= np.min(self.indices[vol]) + 8:
                range_i = range(np.min(self.indices[vol]),np.min(self.indices[vol])+16)
            elif i >= np.max(self.indices[vol]) - 8 and i <= np.max(self.indices[vol]):
                range_i = range(np.max(self.indices[vol]) - 16, np.max(self.indices[vol]))
            self.batches.append(range_i)
            self.vols.append(vol)


        return iter(self.batches)

    def __len__(self):
        return(len(self.n_batches))




