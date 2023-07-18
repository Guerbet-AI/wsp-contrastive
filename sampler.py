import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data.sampler import Sampler
import itertools
from tqdm.auto import tqdm


class CustomSampler(Sampler):

    """A CustomSampler which that provides a batch of volumes with no identical patients.
    :param dataset: a PyTorch Dataset.
    :param batch_size: the batch size used for training.
    :param indices: a dictionary that maps each subjects to the indices present in the dataset.
    :return: a PyTorch sampler."""

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
