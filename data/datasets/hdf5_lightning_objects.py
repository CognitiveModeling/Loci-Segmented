import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import h5py
import numpy as np
from data.datasets.utils import RandomHorizontalFlip, BBoxScaleCrop
import cv2
import time
import math
from turbojpeg import decompress as jpeg_decompress

class HDF5_Dataset(Dataset):
    def __init__(self, hdf5_file_path, crop_size, split=None, max_upscale_factor=1.15, seed=1234):
        
        if not isinstance(crop_size, tuple) and not isinstance(crop_size, list):
            crop_size = (crop_size, crop_size)

        self.filename = hdf5_file_path
        self.split = split
        self.crop_size = crop_size
        self.bbox_crop = BBoxScaleCrop(crop_size, max_upscale_factor=max_upscale_factor)
        self.random_horizontal_flip = RandomHorizontalFlip()

        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

        # Load instance_masks_images into RAM and compute the length of the dataset
        print(f"Loading HDF5 dataset {hdf5_file_path} with crop size {crop_size}", flush=True)
        self.instance_masks_images  = self.hdf5_file["instance_masks_images"][:]
        self.image_instance_indices = self.hdf5_file["image_instance_indices"][:]
        self.dataset_length = len(self.instance_masks_images)
        print(f"Loaded {self.dataset_length} images from HDF5 dataset {hdf5_file_path}", flush=True)
    
        self.bboxes = self.hdf5_file["instance_mask_bboxes"][:]
        print(f"Loaded {len(self.bboxes)} bboxes from HDF5 dataset {hdf5_file_path}", flush=True)

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

        # Use instance_masks_images to find the index of the corresponding RGB and depth images
        image_index = int(self.instance_masks_images[index][0])

        # Load RGB image, depth image, and instance masks
        rgb_image = self.hdf5_file["rgb_images"][image_index]
        instance_mask = self.hdf5_file["instance_masks"][index]
        depth_image = self.hdf5_file["depth_images"][image_index] if self.use_depth else None
        bbox = self.bboxes[index]

        # handle compressed datasets
        if rgb_image.dtype == np.uint8:
            rgb_image     = np.array(jpeg_decompress(rgb_image)).transpose(2, 0, 1).astype(np.float32) / 255.0
            instance_mask = instance_mask.astype(np.float32) / 255.0

            if self.use_depth:
                depth_image = np.expand_dims(cv2.imdecode(depth_image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0, axis=0)

        rgb_image     = th.from_numpy(rgb_image)
        depth_image   = th.from_numpy(depth_image) if self.use_depth else None
        instance_mask = th.from_numpy(instance_mask)

        tensor = th.cat([rgb_image, depth_image, instance_mask] if self.use_depth else [rgb_image, instance_mask], dim=0)

        # applay data augmentation
        tensor, bbox = self.bbox_crop(tensor, bbox)
        tensor, bbox = self.random_horizontal_flip(tensor, bbox)

        if self.use_depth:
            rgb_image, depth_image, instance_mask = tensor.split([3, 1, 1], dim=0)
        else:
            rgb_image, instance_mask = tensor.split([3, 1], dim=0)
            depth_image = th.zeros_like(instance_mask) - 1

        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0 or bbox[0] >= rgb_image.shape[2] or bbox[1] >= rgb_image.shape[1] or bbox[2] >= rgb_image.shape[2] or bbox[3] >= rgb_image.shape[1]:
            print("bbox is out of bounds")
            print(bbox, self.bboxes[index], rgb_image.shape, depth_image.shape, instance_mask.shape)
            debug(self.hdf5_file, rgb_image, depth_image, instance_mask, bbox, index, image_index)
            assert False

        bbox = th.from_numpy(bbox.copy()).float()

        return rgb_image, depth_image, instance_mask

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
