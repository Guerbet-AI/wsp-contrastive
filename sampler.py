import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data.sampler import Sampler
import itertools
from tqdm.auto import tqdm


class CustomSampler(Sampler):

    """A CustomSampler which provides a batch of volumes with no identical patients.
    :param dataset: a PyTorch Dataset.
    :param indices: a dictionary that maps each subjects to the indices present in the dataset.
    :param batch_size: the batch size used for training.
    :param weights: a NumPy array giving the weights of each subject in the dataset according to their class y.
                    Computed automatically in the dataset.py file.
    :return: a PyTorch sampler."""

    def __init__(self,
                 dataset,
                 indices,
                 batch_size: int = 64,
                 k: int = 10000,
                 weights: np.ndarray = None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = k // batch_size
        self.subject_ids = np.unique(dataset.volumes)
        self.weights = weights
        self.indices = indices
        self.depth = 1

    def __iter__(self):

        self.batches = []

        for _ in tqdm(range(self.n_batches)):               # iterate over the number of batches

            batch_idxs = []
            vols = []

            while len(batch_idxs) < self.batch_size:
                # sample in the volumes
                if self.weights is not None:                # sample a volume indice according to the weighting of each class
                    j = np.random.choice(range(len(self.subject_ids)),
                                         p=self.weights / np.sum(self.weights),
                                         size=1)[0]
                else:
                    j = random.choice(range(len(self.subject_ids)))
                vol = self.subject_ids[j]                   # get the associated volume
                if vol not in vols:                         # if the volume is not already in the batch
                    i = random.choice(self.indices[vol])    # sample randomly a slice of this volume
                    batch_idxs.append(i)
                    vols.append(vol)

            self.batches.append(batch_idxs)                 # one the batch is formed, add it to the list of batches

        return iter(list(itertools.chain(*self.batches)))

    def __len__(self):
        return(len(self.n_batches))