import math
import numpy as np

import torch
from torch.utils.data import Dataset, BatchSampler
import torch.distributed as dist
from tqdm import tqdm

class GaussianSequenceSampler():
    def __init__(self, start_index, sequence_length, std_dev, seed=None):
        self.start_index = start_index
        self.sequence_length = sequence_length
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed)
        self.available_indices = list(range(self.sequence_length))
        self.rng.shuffle(self.available_indices)

    def __len__(self):
        return self.sequence_length

    def pop(self):
        # If all indices have been chosen, reset the list of available indices
        if not self.available_indices:
            self.available_indices = list(range(self.sequence_length))
            self.rng.shuffle(self.available_indices)

        # Choose a random index from the available indices
        first_index = self.available_indices.pop()

        if self.std_dev == 0:
            return self.start_index + first_index, self.start_index + first_index

        # sample uniform from whole sequence
        if self.std_dev < 0:
            second_index = self.rng.integers(low=0, high=self.sequence_length)
            return self.start_index + first_index, self.start_index + second_index

        second_index = int(round(self.rng.normal(loc=first_index, scale=self.std_dev)))

        # Ensure the second index is within the sequence
        second_index = max(0, min(self.sequence_length - 1, second_index))
        
        return self.start_index + first_index, self.start_index + second_index


class DistributedBackgroundDatasetSampler():
    def __init__(self, hdf5_dataset, shuffle=True, seed=1234):
        self.dataset   = hdf5_dataset
        self.shuffle   = shuffle
        self.seed      = seed 
        self.epoch     = 0
        self.sequences = [
            GaussianSequenceSampler(
                start_index,
                sequence_length, 
                std_dev = hdf5_dataset.time_std_dev,
                seed=(seed + (i + 1) * 12345)
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

class DistributedBackgroundBatchSampler(BatchSampler):
    def __init__(self, chained_hdf5_dataset, batch_size: int, seed = 1234, shuffle: bool = True, sampler = None) -> None:

        self.cumulative_lengths = chained_hdf5_dataset.cumulative_lengths
        self.length             = len(chained_hdf5_dataset)
        self.samplers           = [DistributedBackgroundDatasetSampler(dataset, shuffle, seed = seed+i) for i, dataset in enumerate(chained_hdf5_dataset.datasets)]
        self.batch_size         = batch_size
        self.seed               = seed + 12121212
        self.shuffle            = shuffle
        self.sampler            = sampler

        self.generate_batches()

    def generate_batches(self):

        self.seed += 1
        np.random.seed(self.seed)
        pseudo_indices = list(np.random.permutation(self.length) if self.shuffle else np.arange(self.length))

        sequences = []
        for i in tqdm(pseudo_indices):
            dataset_index = np.argmax(self.cumulative_lengths > i)
            sequences.append({
                'dataset_index': dataset_index,
                'indices': self.samplers[dataset_index].pop(),
                'seed': np.random.randint(0, 1000000),
                'rank': self.samplers[dataset_index].rank
            })

        self.batches = []
        for i in range(len(sequences) // self.batch_size):
            batch_sequences = sequences[i*self.batch_size:(i + 1)*self.batch_size]
            self.batches.append([])
            for seq in batch_sequences:
                self.batches[-1].append({
                    'dataset_index': seq['dataset_index'],
                    'source_index': seq['indices'][0],
                    'target_index': seq['indices'][1],
                    'seed': seq['seed'],
                    'rank': seq['rank'],
                })

    def __iter__(self):
        for batch in self.batches:
            yield batch

        self.generate_batches()

    def __len__(self) -> int:
        return len(self.batches)

