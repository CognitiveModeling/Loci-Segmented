import math
import numpy as np

import torch
from torch.utils.data import Dataset, BatchSampler
import torch.distributed as dist

class SequnceSampler():
    """
    A class for generating and sampling random, non-overlapping subsets from a given sequence.
    
    The sequence is defined by a starting index and a sequence length, and the size of the subsets
    is defined by the subset_length parameter. Each new batch of subsets begins from a random offset,
    and the subsets are returned in a randomized order. Once all subsets have been sampled,
    a new batch of subsets is generated with a new random offset.

    Attributes
    ----------
    start_index : int
        The starting index of the sequence.
        
    sequence_length : int
        The length of the sequence. The sequence will contain integers from start_index to
        start_index + sequence_length - 1 (inclusive).
        
    subset_length : int
        The length of the subsets that will be sampled from the sequence. The sequence's length
        should be a multiple of subset_length for uniform sampling.
        
    seed : int
        The seed for numpy's random number generator. This is used for generating random offsets
        and shuffling the subsets.
        
    shuffle : bool
        Unused in the provided code.

    Methods
    -------
    __len__():
        Returns the number of subsets that the sequence will be divided into.
        
    generate_random_offsets():
        Generates a permutation of numbers from 0 to subset_length - 1 (inclusive) as possible offsets.
        
    generate_subsets():
        Generates subsets of the sequence by slicing it at intervals of subset_length, starting from
        a random offset. It then shuffles these subsets.
        
    pop():
        Returns and removes the last subset in self.subsets. If self.subsets is empty, generates a
        new set of subsets before returning a subset.
    """
    def __init__(self, start_index, sequence_length, subset_length, seed, shuffle):
        self.start_index     = start_index
        self.sequence_length = sequence_length
        self.subset_length   = subset_length
        self.seed            = seed
        self.length          = sequence_length // subset_length
        
        if self.length == 0:
            raise ValueError("Sequence length must be a multiple of subset length")
        
        if self.sequence_length != self.subset_length:
            self.generate_random_offsets()
            self.generate_subsets()

    def __len__(self):
        return self.length

    def generate_random_offsets(self):
        self.seed += 1
        np.random.seed(self.seed)
        self.offsets = list(np.random.permutation(min(self.subset_length, self.sequence_length - self.subset_length)))

    def generate_subsets(self):
        if len(self.offsets) == 0:
            self.generate_random_offsets()

        offset = self.offsets.pop()
        self.subsets = []
        for i in range(offset, self.sequence_length - self.subset_length, self.subset_length):
            self.subsets.append(np.arange(self.start_index + i, self.start_index + i + self.subset_length))

            for i in self.subsets[-1]:
                assert i >= self.start_index and i < self.start_index + self.sequence_length

        self.seed += 1
        np.random.seed(self.seed)
        self.subsets = list(np.random.permutation(self.subsets))

    def pop(self):
        if self.sequence_length == self.subset_length:
            return np.arange(self.start_index, self.start_index + self.sequence_length)

        if len(self.subsets) == 0:
            self.generate_subsets()

        return self.subsets.pop()

class DistributedSequenceSampler():
    """
    A class for managing distributed data loading from an HDF5 dataset, where the data is being loaded across 
    multiple processes or nodes. It uses instances of the `SequnceSampler` class to generate and shuffle 
    subsets of data for each node to load.

    Attributes
    ----------
    hdf5_dataset : HDF5 dataset object
        The HDF5 dataset to load data from.

    subset_length : int
        The length of the subsets to be sampled from each sequence in the dataset.

    shuffle : bool, default=True
        Whether to shuffle the indices of the data before distributing them across the nodes.

    seed : int, default=1234
        The seed for numpy's random number generator. Used for shuffling indices and generating subsets.

    Methods
    -------
    distribut_indices():
        Distributes indices across different nodes. At the start of each new epoch, it generates a new set of 
        indices by either permuting the total number of raw samples (if `shuffle` is True) or creating a range 
        from 0 to `total_size`. Each node then takes a specific slice of these indices according to its rank.

    __len__():
        Returns the number of samples each node is responsible for handling.

    pop():
        Pops an index from the current node's slice of indices. If the slice is empty, it redistributes indices 
        across the nodes. The popped index is used to find the corresponding `SequnceSampler` instance, from 
        which a subset is then popped and returned.
    """
    def __init__(self, hdf5_dataset, subset_length, shuffle=True, seed=1234):
        self.dataset   = hdf5_dataset
        self.shuffle   = shuffle
        self.seed      = seed 
        self.epoch     = 0
        self.sequences = [
            SequnceSampler(
                start_index, 
                sequence_length, 
                subset_length, 
                seed + (i + 1) * 12345, 
                shuffle
            ) for i, (start_index, sequence_length) in enumerate(hdf5_dataset.sequence_indices)
        ]
        self.raw_length = sum([len(s) for s in self.sequences])
        self.cumulative_lengths = np.cumsum([len(s) for s in self.sequences])

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))

        if self.raw_length % self.num_replicas != 0: 
            self.num_samples = math.ceil((self.raw_length - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.raw_length / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.distribute_indices()

    def distribute_indices(self):
        self.epoch += 1
        np.random.seed(self.seed + self.epoch)
        indices = np.random.permutation(self.raw_length) if self.shuffle else np.arange(self.total_size)

        self.indices = list(indices[self.rank*self.num_samples:(self.rank + 1)*self.num_samples])

    def __len__(self):
        return self.num_samples

    def pop(self):
        if len(self.indices) == 0:
            self.distribute_indices()
            print(f"rank {self.rank} is out of indices, re-distributing indices (dataset size: {len(self.sequences)}) ({self.dataset.filename})")

        index = self.indices.pop()
        sampler_index = np.argmax(self.cumulative_lengths > index)
        return self.sequences[sampler_index].pop()
