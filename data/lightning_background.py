from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.configuration import Configuration
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
import h5py
import numpy as np
import torch
from data.datasets.hdf5_lightning_background import HDF5_Dataset, ChainedHDF5_Dataset
from data.sampler.background_sampler import DistributedBackgroundBatchSampler

class LociBackgroundDataModule(LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg
        self.trainset = ChainedHDF5_Dataset(
            [HDF5_Dataset(
                hdf5_file_path          = d.path, 
                crop_size               = cfg.model.input_size, 
                split                   = "train" if d.split else None,
                use_depth_weighting     = d.depth_weighting, 
                time_sample_std_dev     = d.time_std_dev, 
                static_background       = d.static,
                uncertainty_regularizer = d.uncertainty_regularizer,
                rgb_factor              = d.rgb_loss_factor,
                rgb_warmup              = d.rgb_warmup,
                depth_warmup            = d.depth_warmup,
                color_input             = d.color_input,
            ) for d in cfg.data.train],
            [d.weight for d in cfg.data.train],
        )
        self.valset = ChainedHDF5_Dataset(
            [HDF5_Dataset(
                hdf5_file_path          = d.path, 
                crop_size               = cfg.model.input_size, 
                split                   = "val" if d.split else None,
                use_depth_weighting     = d.depth_weighting, 
                time_sample_std_dev     = d.time_std_dev, 
                static_background       = d.static,
                uncertainty_regularizer = d.uncertainty_regularizer,
                rgb_factor              = d.rgb_loss_factor,
                rgb_warmup              = d.rgb_warmup,
                depth_warmup            = d.depth_warmup,
                color_input             = d.color_input,
            ) for d in cfg.data.val],
            [d.weight for d in cfg.data.val],
        )
        self.testset = ChainedHDF5_Dataset(
            [HDF5_Dataset(
                hdf5_file_path          = d.path, 
                crop_size               = cfg.model.input_size, 
                split                   = "test" if d.split else None,
                use_depth_weighting     = d.depth_weighting, 
                time_sample_std_dev     = d.time_std_dev, 
                static_background       = d.static,
                uncertainty_regularizer = d.uncertainty_regularizer,
                rgb_factor              = d.rgb_loss_factor,
                rgb_warmup              = d.rgb_warmup,
                depth_warmup            = d.depth_warmup,
                color_input             = d.color_input,
            ) for d in cfg.data.test],
            [d.weight for d in cfg.data.test],
        )

        self.batch_size = self.cfg.model.batch_size

    def train_dataloader(self):
        sampler = DistributedBackgroundBatchSampler(self.trainset, self.cfg.model.batch_size, seed=self.cfg.seed)

        return DataLoader(
            self.trainset,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            batch_sampler=sampler,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=True,
        )

    def val_dataloader(self):
        sampler = DistributedBackgroundBatchSampler(self.valset, self.cfg.model.batch_size, shuffle=True)

        return DataLoader(
            self.valset, 
            pin_memory=True, 
            num_workers=self.cfg.num_workers, 
            batch_sampler=sampler,
            prefetch_factor=self.cfg.prefetch_factor, 
            persistent_workers=True
        )

    def test_dataloader(self):
        sampler = DistributedBackgroundBatchSampler(self.testset, self.cfg.model.batch_size, shuffle=False)

        return DataLoader(
            self.testset, 
            pin_memory=True, 
            num_workers=self.cfg.num_workers, 
            batch_sampler=sampler,
            prefetch_factor=self.cfg.prefetch_factor, 
            persistent_workers=True
        )
