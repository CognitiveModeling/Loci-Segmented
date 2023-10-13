import math
import numpy as np

import torch
from torch.utils.data import Dataset, BatchSampler
import torch.distributed as dist
from data.sampler.sequence_sampler import DistributedSequenceSampler


class DistributedLociBatchSampler(BatchSampler):
    def __init__(self, chained_hdf5_dataset, sequence_length, batch_size: int, num_time_steps: int, teacher_forcing: int, seed = 1234, shuffle: bool = True, sampler = None) -> None:

        self.cumulative_lengths = chained_hdf5_dataset.cumulative_lengths
        self.lenght             = len(chained_hdf5_dataset)
        self.samplers           = [DistributedSequenceSampler(dataset, sequence_length, shuffle, seed = seed+i) for i, dataset in enumerate(chained_hdf5_dataset.datasets)]
        self.sequence_length    = sequence_length
        self.num_time_steps     = num_time_steps
        self.batch_size         = batch_size
        self.seed               = seed + 12121212
        self.shuffle            = shuffle
        self.sampler            = sampler
        self.teacher_forcing    = (teacher_forcing // num_time_steps) * num_time_steps + 1 if sequence_length > 1 else teacher_forcing

        assert self.sequence_length % self.num_time_steps == 0 or self.sequence_length == 1, "sequence_length must be divisible by num_time_steps"
        
        self.generate_batches()

    def generate_batches(self):

        self.seed += 1
        np.random.seed(self.seed)
        pseudo_indices = list(np.random.permutation(self.lenght) if self.shuffle else np.arange(self.lenght))

        # selecte every sequence_length'th index
        pseudo_indices = pseudo_indices[::self.sequence_length]

        sequences = []
        for i in pseudo_indices:
            dataset_index = np.argmax(self.cumulative_lengths > i)
            sequences.append({
                'dataset_index': dataset_index,
                'sequence_indices': self.samplers[dataset_index].pop(),
                'seed': np.random.randint(0, 1000000),
                'rank': self.samplers[dataset_index].rank
            })

            # check wether sequence indices are increased one by one
            assert np.all(np.diff(sequences[-1]['sequence_indices']) == 1), "sequence indices must be increased one by one"

        self.batches = []
        for i in range(len(sequences) // self.batch_size):
            batch_sequences = sequences[i*self.batch_size:(i + 1)*self.batch_size]
            if self.sequence_length == 1:
                self.batches.append([])
                for seq in batch_sequences:
                    self.batches[-1].append({
                        'dataset_index': seq['dataset_index'],
                        'sequence_index': seq['sequence_indices'][0],
                        'sequence_length': self.num_time_steps,
                        'seed': seq['seed'],
                        'rank': seq['rank'],
                        'start_time': -self.teacher_forcing,
                    })

            else:
                for t in range(self.sequence_length // self.num_time_steps):
                    if t == 0:
                        for tt in range(self.teacher_forcing // self.num_time_steps):
                            self.batches.append([])
                            for seq in batch_sequences:
                                self.batches[-1].append({
                                    'dataset_index': seq['dataset_index'],
                                    'sequence_index': seq['sequence_indices'][0],
                                    'sequence_length': self.num_time_steps + (1 if tt == 0 else 0),
                                    'seed': seq['seed'],
                                    'rank': seq['rank'],
                                    'start_time': -self.teacher_forcing + tt * self.num_time_steps + (0 if tt == 0 else 1),
                                })

                    self.batches.append([])
                    for seq in batch_sequences:
                        self.batches[-1].append({
                            'dataset_index': seq['dataset_index'],
                            'sequence_index': seq['sequence_indices'][t*self.num_time_steps],
                            'sequence_length': self.num_time_steps,
                            'seed': seq['seed'],
                            'rank': seq['rank'],
                            'start_time': t * self.num_time_steps,
                        })

    def __iter__(self):
        for batch in self.batches:
            yield batch

        self.generate_batches()

    def __len__(self) -> int:
        return len(self.batches)

