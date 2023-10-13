from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.configuration import Configuration
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
import h5py
import numpy as np
import torch
from data.datasets.hdf5_lightning_objects import HDF5_Dataset, ChainedHDF5_Dataset
from data.sampler.object_sampler import DistributedObjectSampler

class LociPretrainerDataModule(LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg
        self.trainset = ChainedHDF5_Dataset(
            [HDF5_Dataset(d.path, cfg.model.input_size, "train" if d.split else None, seed=cfg.seed) for d in cfg.data.train],
            [d.weight for d in cfg.data.train],
        )
        self.valset = ChainedHDF5_Dataset(
            [HDF5_Dataset(d.path, cfg.model.input_size, "val" if d.split else None) for d in cfg.data.val],
            [d.weight for d in cfg.data.val],
        )
        self.testset = ChainedHDF5_Dataset(
            [HDF5_Dataset(d.path, cfg.model.input_size, "test" if d.split else None) for d in cfg.data.test],
            [d.weight for d in cfg.data.test],
        )

        self.batch_size = self.cfg.model.batch_size

    def train_dataloader(self):
        sampler = DistributedObjectSampler(self.trainset, shuffle=True, seed=self.cfg.seed)

        return DataLoader(
            self.trainset,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=True,
        )

    def val_dataloader(self):
        sampler = DistributedObjectSampler(self.valset, shuffle=True)

        return DataLoader(
            self.valset, 
            pin_memory=True, 
            num_workers=self.cfg.num_workers, 
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True, 
            prefetch_factor=self.cfg.prefetch_factor, 
            persistent_workers=True
        )

    def test_dataloader(self):
        sampler = DistributedObjectSampler(self.testset, shuffle=False)

        return DataLoader(
            self.testset, 
            pin_memory=True, 
            num_workers=self.cfg.num_workers, 
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True, 
            prefetch_factor=self.cfg.prefetch_factor, 
            persistent_workers=True
        )
