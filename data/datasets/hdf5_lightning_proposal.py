import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import h5py
import numpy as np
from data.datasets.utils import RandomHorizontalFlip, ScaleCrop
import cv2
import time
import math
from turbojpeg import decompress as jpeg_decompress
from einops import repeat

class HDF5_Dataset(Dataset):
    def __init__(self, hdf5_file_path, crop_size, split=None, max_upscale_factor=1.15, seed=1234, max_num_mask_per_image=32):
        
        if not isinstance(crop_size, tuple) and not isinstance(crop_size, list):
            crop_size = (crop_size, crop_size)

        self.max_num_mask_per_image = max_num_mask_per_image
        self.filename = hdf5_file_path
        self.split = split
        self.crop_size = crop_size
        self.random_crop = ScaleCrop(crop_size)
        self.random_horizontal_flip = RandomHorizontalFlip(flip_dim=3)

        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

        # Load instance_masks_images into RAM and compute the length of the dataset
        print(f"Loading HDF5 dataset {hdf5_file_path}", flush=True)
        self.image_instance_indices = self.hdf5_file["image_instance_indices"][:]
        self.dataset_length = len(self.image_instance_indices)
        if len(self.image_instance_indices) == 0:
            self.dataset_length = self.hdf5_file["rgb_images"].shape[0]
        print(f"Loaded {self.dataset_length} images from HDF5 dataset {hdf5_file_path}", flush=True)
    
        self.use_depth = "depth_images" in self.hdf5_file and self.hdf5_file["depth_images"].shape[0] > 1

        self.hdf5_file.close()
        self.hdf5_file = None

        np.random.seed(seed)
        self.indices = np.arange(self.dataset_length)
        np.random.shuffle(self.indices)

        print(f"Loaded HDF5 dataset {hdf5_file_path} with size {self.dataset_length}", flush=True)

    def __len__(self):
        if self.split is not None:
            if self.split == "train":
                return int(self.dataset_length * 0.8)
            else:
                return int(self.dataset_length * 0.2)

        return self.dataset_length

    def __getitem__(self, index):

        if self.split is not None:
            if self.split == "train":
                index = self.indices[index]
            else:
                index = self.indices[index + int(self.dataset_length * 0.8)]

        # Open the HDF5 file if it is not already open
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

        # Load RGB and depth image
        rgb_image   = self.hdf5_file["rgb_images"][index]
        depth_image = self.hdf5_file["depth_images"][index] if self.use_depth else None

        if len(self.image_instance_indices) != 0:
            mask_index_start, mask_index_length = self.image_instance_indices[index]
            instance_masks = self.hdf5_file["instance_masks"][mask_index_start:mask_index_start + mask_index_length][:,0]

            # sort mask by size
            mask_sizes     = np.sum(instance_masks, axis=(1, 2))
            mask_indices   = np.argsort(mask_sizes)[::-1]
            instance_masks = (instance_masks[mask_indices] / 255.0).astype(np.float32)

        # handle compressed datasets
        if rgb_image.dtype == np.uint8:
            rgb_image      = np.array(jpeg_decompress(rgb_image)).transpose(2, 0, 1).astype(np.float32) / 255.0

            if self.use_depth:
                depth_image = np.expand_dims(cv2.imdecode(depth_image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0, axis=0)

        if len(self.image_instance_indices) == 0:
            instance_masks = np.zeros((0, rgb_image.shape[1], rgb_image.shape[2]), dtype=np.uint8)

        rgb_image      = th.from_numpy(rgb_image)
        depth_image    = th.from_numpy(depth_image) if self.use_depth else None
        instance_masks = th.from_numpy(instance_masks)

        tensor = th.cat([rgb_image, depth_image, instance_masks] if self.use_depth else [rgb_image, instance_masks], dim=0)

        # applay data augmentation
        tensor = self.random_crop(tensor.unsqueeze(0))
        tensor = self.random_horizontal_flip(tensor)[0].squeeze(0)

        if self.use_depth:
            rgb_image, depth_image, instance_masks = tensor.split([3, 1, instance_masks.shape[0]], dim=0)
        else:
            rgb_image, instance_masks = tensor.split([3, instance_masks.shape[0]], dim=0)
            depth_image = th.zeros_like(tensor[:1]) - 1

        mask_index_length = min(instance_masks.shape[0], self.max_num_mask_per_image)
        _instance_masks = th.zeros((self.max_num_mask_per_image, *self.crop_size), dtype=th.float32)
        _instance_masks[:mask_index_length] = instance_masks[:mask_index_length]
        instance_masks = _instance_masks

        return rgb_image, depth_image, instance_masks

class ChainedHDF5_Dataset(Dataset):
    def __init__(self, hdf5_datasets, weights):
        self.datasets = hdf5_datasets
        self.weights  = weights / np.sum(weights)
        self.dataset_offset = 1000**3

        total_length = sum([len(d) for d in self.datasets])
        self.lenght  = sum([int(total_length * w) for w in self.weights])

        print(f"dataset size: {self.lenght}")
        for d, w in zip(self.datasets, self.weights):
            print(f"resampling dataset {len(d):10d}|{100*len(d)/total_length:.1f}% -> {int(total_length * w):10d}|{100*w:.1f}% ({d.filename})")

        self.cumulative_lengths = np.cumsum([int(total_length * w) for w in self.weights])

    def __len__(self):
        return self.lenght

    def __getitem__(self, combined_index):

        dataset_index = combined_index // self.dataset_offset
        index         = combined_index % self.dataset_offset

        return self.datasets[dataset_index][index]
