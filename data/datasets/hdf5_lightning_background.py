import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
import h5py
import numpy as np
from data.datasets.utils import RandomHorizontalFlip, ScaleCrop
import cv2
from turbojpeg import decompress as jpeg_decompress

class HDF5_Dataset(Dataset):
    def __init__(
        self, 
        hdf5_file_path, 
        crop_size, 
        split=None, 
        use_depth_weighting=False, 
        time_sample_std_dev=0, 
        static_background=False,
        uncertainty_regularizer=0.01,
        rgb_factor=1.0,
        rgb_warmup=0.0,
        depth_warmup=0.0,
        color_input=0,
    ):
        
        if not isinstance(crop_size, tuple) and not isinstance(crop_size, list):
            crop_size = (crop_size, crop_size)

        self.filename = hdf5_file_path
        self.random_crop = ScaleCrop(crop_size)
        self.random_horizontal_flip = RandomHorizontalFlip(flip_dim=3)
        self.time_std_dev = time_sample_std_dev
        self.static_background = th.tensor(static_background).float()
        self.uncertainty_regularizer = th.tensor(uncertainty_regularizer).float()
        self.depth_weighting = th.tensor(use_depth_weighting).float()
        self.rgb_factor = th.tensor(rgb_factor).float()
        self.rgb_warmup = th.tensor(rgb_warmup).float()
        self.depth_warmup = th.tensor(depth_warmup).float()
        self.color_input = th.tensor(color_input).float()

        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

        self.sequence_indices = self.hdf5_file["sequence_indices"][:]
        if split is not None:
            if split == "train":
                self.sequence_indices = self.sequence_indices[:int(len(self.sequence_indices) * 0.8)]
            else:
                self.sequence_indices = self.sequence_indices[int(len(self.sequence_indices) * 0.8):]

        self.dataset_length = sum([seq[1] for seq in self.sequence_indices])

        self.use_depth    = "depth_images" in self.hdf5_file and self.hdf5_file["depth_images"].shape[0] > 1
        self.use_fg_masks = "foreground_mask" in self.hdf5_file and self.hdf5_file["foreground_mask"].shape[0] > 1
        self.use_fg_masks = self.use_fg_masks and np.sum(self.hdf5_file["foreground_mask"][0:100]) > 0

        self.hdf5_file.close()
        self.hdf5_file = None

        print(f"Loaded HDF5 dataset {hdf5_file_path} with size {self.dataset_length}")

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, sample):

        source_index = sample['source_index']
        target_index = sample['target_index']
        seed         = sample['seed']

        delta_t = th.tensor([target_index - source_index], dtype=th.float32)
        delta_t = delta_t * (1 - self.static_background)

        # Open the HDF5 file if it is not already open
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

        rgb_source_image       = self.hdf5_file["rgb_images"][source_index]
        depth_source_image     = self.hdf5_file["depth_images"][source_index] if self.use_depth else np.zeros((1, *rgb_source_image.shape[1:]))
        foreground_source_mask = self.hdf5_file["foreground_mask"][source_index] if self.use_fg_masks else np.zeros((1, *rgb_source_image.shape[1:]))

        rgb_target_image       = self.hdf5_file["rgb_images"][target_index]
        depth_target_image     = self.hdf5_file["depth_images"][target_index] if self.use_depth else np.zeros((1, *rgb_target_image.shape[1:]))
        foreground_target_mask = self.hdf5_file["foreground_mask"][target_index] if self.use_fg_masks else np.zeros((1, *rgb_target_image.shape[1:]))

        # handle compressed datasets
        if rgb_source_image.dtype == np.uint8:
            rgb_source_image = np.array(jpeg_decompress(rgb_source_image)).transpose(2, 0, 1).astype(np.float32) / 255.0
            rgb_target_image = np.array(jpeg_decompress(rgb_target_image)).transpose(2, 0, 1).astype(np.float32) / 255.0

            if self.use_depth:
                depth_source_image = np.expand_dims(cv2.imdecode(depth_source_image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0, axis=0)
                depth_target_image = np.expand_dims(cv2.imdecode(depth_target_image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0, axis=0)

            if self.use_fg_masks:
                foreground_source_mask = foreground_source_mask.astype(np.float32) / 255.0
                foreground_target_mask = foreground_target_mask.astype(np.float32) / 255.0

        rgb_source_image       = th.from_numpy(rgb_source_image)
        depth_source_image     = th.from_numpy(depth_source_image) if self.use_depth else None
        foreground_source_mask = th.from_numpy(foreground_source_mask) if self.use_fg_masks else None

        rgb_target_image       = th.from_numpy(rgb_target_image)
        depth_target_image     = th.from_numpy(depth_target_image) if self.use_depth else None
        foreground_target_mask = th.from_numpy(foreground_target_mask) if self.use_fg_masks else None

        tensor = [rgb_source_image, rgb_target_image]
        if self.use_depth:
            tensor.append(depth_source_image)
            tensor.append(depth_target_image)
        if self.use_fg_masks:
            tensor.append(foreground_source_mask)
            tensor.append(foreground_target_mask)

        tensor = th.cat(tensor, dim=0).unsqueeze(0)

        # applay data augmentation
        tensor = self.random_crop(tensor, seed=seed)
        tensor = self.random_horizontal_flip(tensor, seed=seed)[0]

        if self.use_depth and self.use_fg_masks:
            rgb_source_image, rgb_target_image, depth_source_image, depth_target_image, foreground_source_mask, foreground_target_mask = tensor[0].split([3, 3, 1, 1, 1, 1], dim=0)
        elif self.use_depth:
            rgb_source_image, rgb_target_image, depth_source_image, depth_target_image = tensor[0].split([3, 3, 1, 1], dim=0)
            foreground_source_mask = th.zeros_like(depth_source_image)
            foreground_target_mask = th.zeros_like(depth_target_image)
        elif self.use_fg_masks:
            rgb_source_image, rgb_target_image, foreground_source_mask, foreground_target_mask = tensor[0].split([3, 3, 1, 1], dim=0)
            depth_source_image = th.zeros_like(foreground_source_mask) - 1
            depth_target_image = th.zeros_like(foreground_target_mask) - 1
        else:
            rgb_source_image, rgb_target_image = tensor[0].split([3, 3], dim=0)
            depth_source_image = th.zeros_like(rgb_source_image[:1]) - 1
            depth_target_image = th.zeros_like(rgb_target_image[:1]) - 1
            foreground_source_mask = th.zeros_like(rgb_source_image[:1])
            foreground_target_mask = th.zeros_like(rgb_target_image[:1])

        return (
            rgb_source_image, 
            depth_source_image, 
            foreground_source_mask, 
            rgb_target_image, 
            depth_target_image, 
            foreground_target_mask, 
            self.use_depth, 
            self.use_fg_masks, 
            delta_t,
            self.uncertainty_regularizer,
            self.depth_weighting,
            self.rgb_factor,
            self.rgb_warmup,
            self.depth_warmup,
            self.color_input,
        )

class ChainedHDF5_Dataset(Dataset):
    def __init__(self, hdf5_datasets, weights):
        self.datasets = hdf5_datasets
        self.weights  = weights / np.sum(weights)

        total_length = sum([len(d) for d in self.datasets])
        self.lenght  = sum([int(total_length * w) for w in self.weights])
        print(f"dataset size: {self.lenght}, total length: {total_length}")

        print(f"dataset size: {self.lenght}")
        for d, w in zip(self.datasets, self.weights):
            print(f"resampling dataset {len(d):10d}|{100*len(d)/total_length:.1f}% -> {int(total_length * w):10d}|{100*w:.1f}% ({d.filename})")

        self.cumulative_lengths = np.cumsum([int(total_length * w) for w in self.weights])
        assert self.lenght == self.cumulative_lengths[-1]

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, sample):
        return self.datasets[sample['dataset_index']][sample]
