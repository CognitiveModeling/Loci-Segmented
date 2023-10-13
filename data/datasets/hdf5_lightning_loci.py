import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
import h5py
import numpy as np
from data.datasets.utils import RandomHorizontalFlip, ScaleCrop
import cv2
from turbojpeg import decompress as jpeg_decompress

class HDF5_Dataset(Dataset):
    def __init__(self, hdf5_file_path, crop_size, split=None, load_fg_mask=False, data_augmentation=True, load_masks=False, max_masks = 32):
        
        if not isinstance(crop_size, tuple) and not isinstance(crop_size, list):
            crop_size = (crop_size, crop_size)
        
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.load_fg_mask = load_fg_mask
        self.max_masks = max_masks
        self.load_masks = load_masks
        self.filename = hdf5_file_path
        self.random_crop = ScaleCrop(crop_size)
        self.random_horizontal_flip = RandomHorizontalFlip(flip_dim=3)

        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

        self.sequence_indices = self.hdf5_file["sequence_indices"][:] if "sequence_indices" in self.hdf5_file else np.array([[0, self.hdf5_file["rgb_images"].shape[0]]])
        self.image_instance_indices = self.hdf5_file["image_instance_indices"][:]
        if split is not None:
            if split == "train":
                self.sequence_indices = self.sequence_indices[:int(len(self.sequence_indices) * 0.8)]
            else:
                self.sequence_indices = self.sequence_indices[int(len(self.sequence_indices) * 0.8):]

        self.dataset_length = sum([seq[1] for seq in self.sequence_indices])

        self.use_depth    = "depth_images" in self.hdf5_file and self.hdf5_file["depth_images"].shape[0] > 1
        self.has_fg_masks = "foreground_mask" in self.hdf5_file and self.hdf5_file["foreground_mask"].shape[0] > 1
        self.has_instace_masks = "instance_masks" in self.hdf5_file and self.hdf5_file["instance_masks"].shape[0] > 1

        self.hdf5_file.close()
        self.hdf5_file = None

        print(f"Loaded HDF5 dataset {hdf5_file_path} with size {self.dataset_length}")

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, sample):

        index      = sample['sequence_index']
        length     = sample['sequence_length']
        seed       = sample['seed']
        start_time = sample['start_time']

        # Open the HDF5 file if it is not already open
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

        rgb_images   = self.hdf5_file["rgb_images"][index:index + (length if start_time >= 0 else 1)]
        depth_images = self.hdf5_file["depth_images"][index:index + (length if start_time >= 0 else 1)] if self.use_depth else None
        time_steps   = np.arange(start_time, start_time+length)

        compressed_dataset = False
        if rgb_images[0].dtype == np.uint8:
            compressed_dataset = True
            images = []
            depths = []
            for i in range(len(rgb_images)):
                images.append(np.array(jpeg_decompress(rgb_images[i])).transpose(2, 0, 1).astype(np.float32) / 255.0)

            rgb_images   = np.stack(images)

            if self.use_depth:
                for i in range(len(depth_images)):
                    depths.append(np.expand_dims(cv2.imdecode(depth_images[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0, axis=0))

                depth_images = np.stack(depths)

        rgb_images   = th.from_numpy(rgb_images)
        depth_images = th.from_numpy(depth_images) if self.use_depth else None

        if self.load_fg_mask:
            if self.has_fg_masks:
                fg_masks = self.hdf5_file["foreground_mask"][index:index + (length if start_time >= 0 else 1)]
                fg_masks = th.from_numpy(fg_masks) / 255.0
            else:
                fg_masks = th.zeros((length if start_time >= 0 else 1, 1, rgb_images.shape[2], rgb_images.shape[3]))

        if self.load_masks:
            mask_index_start  = self.image_instance_indices[index:index + (length if start_time >= 0 else 1), 0]
            mask_index_length = self.image_instance_indices[index:index + (length if start_time >= 0 else 1), 1]

            instance_masks = th.zeros((length if start_time >= 0 else 1, self.max_masks, rgb_images.shape[2], rgb_images.shape[3]))
            if self.has_instace_masks:
                for t in range(length if start_time >= 0 else 1):
                    instance_masks[t, :mask_index_length[t]] = th.from_numpy(self.hdf5_file["instance_masks"][mask_index_start[t]:mask_index_start[t] + min(mask_index_length[t], self.max_masks)][:,0])

                instance_masks = instance_masks / 255.0
            

        tensor = th.cat([rgb_images, depth_images], dim=1) if self.use_depth else rgb_images
        tensor = th.cat([tensor, fg_masks], dim=1) if self.load_fg_mask else tensor
        tensor = th.cat([tensor, instance_masks], dim=1) if self.load_masks else tensor

        # applay data augmentation
        if self.data_augmentation:
            tensor = self.random_crop(tensor, seed=seed)
            tensor = self.random_horizontal_flip(tensor, seed=seed)[0]
        elif tensor.shape[2] != self.crop_size[0] or tensor.shape[3] != self.crop_size[1]:
            tensor = th.nn.functional.interpolate(tensor, size=self.crop_size, mode="bilinear", align_corners=False)

        if self.use_depth:
            if self.load_fg_mask:
                if self.load_masks:
                    rgb_images, depth_images, fg_masks, instance_masks = th.split(tensor, [3, 1, 1, self.max_masks], dim=1)
                else:
                    rgb_images, depth_images, fg_masks = th.split(tensor, [3, 1, 1], dim=1)
            else:
                if self.load_masks:
                    rgb_images, depth_images, instance_masks = th.split(tensor, [3, 1, self.max_masks], dim=1)
                else:
                    rgb_images, depth_images = th.split(tensor, [3, 1], dim=1)
        else:
            depth_images = th.zeros_like(tensor[:,:1]) - 1
            if self.load_fg_mask:
                if self.load_masks:
                    rgb_images, fg_masks, instance_masks = th.split(tensor, [3, 1, self.max_masks], dim=1)
                else:
                    rgb_images, fg_masks = th.split(tensor, [3, 1], dim=1)
            else:
                if self.load_masks:
                    rgb_images, instance_masks = th.split(tensor, [3, self.max_masks], dim=1)
                else:
                    rgb_images = tensor

        if self.load_fg_mask:
            if self.load_masks:
                return rgb_images, depth_images, time_steps, self.use_depth, fg_masks, th.from_numpy(instance_masks.numpy())

            return rgb_images, depth_images, time_steps, self.use_depth, fg_masks

        if self.load_masks:
            return rgb_images, depth_images, time_steps, self.use_depth, instance_masks

        return rgb_images, depth_images, time_steps, self.use_depth

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

