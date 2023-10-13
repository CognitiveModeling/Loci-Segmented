import numpy as np
import cv2
import os
import random
import lz4.frame
import io
import math
import torch.nn.functional as F
import torch as th

class RandomHorizontalFlip:
    def __init__(self, flip_probability=0.5, flip_dim=2):
        self.flip_probability = flip_probability
        self.flip_dim = flip_dim

    def __call__(self, tensor, bbox=None, seed=None):

        if seed is not None:
            np.random.seed(seed)

        do_flip = np.random.rand() < self.flip_probability

        if do_flip:
            tensor = th.flip(tensor, dims=(self.flip_dim,))
            
            if bbox is not None:
                _, H, W = tensor.shape
                bbox[0] = W - bbox[0] - 1
                bbox[2] = W - bbox[2] - 1

        return tensor, bbox

class BBoxScaleCrop:
    def __init__(self, crop_size, max_upscale_factor=1.15):
        self.crop_height, self.crop_width = crop_size
        self.max_upscale_factor = max_upscale_factor

    def get_random_crop_coords(self, bbox, img_height, img_width, crop_height, crop_width):
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox.astype(np.int32)

        crop_y_min = min(max(bbox_y_max - crop_height + 1, 0), img_height - crop_height)
        crop_y_max = max(min(bbox_y_min, img_height - crop_height), crop_y_min)
        crop_x_min = min(max(bbox_x_max - crop_width + 1, 0), img_width - crop_width)
        crop_x_max = max(min(bbox_x_min, img_width - crop_width), crop_x_min)

        # check wether bbox is bigger than crop size
        if bbox_x_max - bbox_x_min > crop_width:
            crop_x_min = bbox_x_min
            crop_x_max = bbox_x_max - crop_width + 1

        if bbox_y_max - bbox_y_min > crop_height:
            crop_y_min = bbox_y_min
            crop_y_max = bbox_y_max - crop_height + 1

        rand_y = np.random.randint(0, crop_y_max - crop_y_min) if crop_y_max - crop_y_min > 0 else 0
        rand_x = np.random.randint(0, crop_x_max - crop_x_min) if crop_x_max - crop_x_min > 0 else 0

        y1 = int(rand_y + crop_y_min)
        x1 = int(rand_x + crop_x_min)

        return y1, x1

    def scale_crop(self, tensor, y1, x1, crop_height, crop_width):
        cropped_tensor = tensor[:, y1:y1 + crop_height, x1:x1 + crop_width] if tensor.shape[1] >= crop_height and tensor.shape[2] >= crop_width else tensor
        if self.crop_height == crop_height and self.crop_width == crop_width:
            return cropped_tensor

        return F.interpolate(cropped_tensor.unsqueeze(0), size=(self.crop_height, self.crop_width), mode='bilinear', align_corners=False)[0]

    def compute_new_bbox(self, bbox, crop_coords, crop_height, crop_width):
        # Get the original bounding box coordinates
        x_min, y_min, x_max, y_max = bbox.astype(np.int32)

        # Get the crop coordinates
        y1, x1 = crop_coords

        # Calculate the new bounding box coordinates after cropping
        new_x_min = max(x_min - x1, 0)
        new_y_min = max(y_min - y1, 0)
        new_x_max = min(x_max - x1, crop_width)
        new_y_max = min(y_max - y1, crop_height)

        # Calculate the scale factors for the new bounding box
        scale_factor_h = self.crop_height / crop_height
        scale_factor_w = self.crop_width / crop_width

        # Scale the new bounding box coordinates
        scaled_new_x_min = int(new_x_min * scale_factor_w)
        scaled_new_y_min = int(new_y_min * scale_factor_h)
        scaled_new_x_max = int(new_x_max * scale_factor_w)
        scaled_new_y_max = int(new_y_max * scale_factor_h)

        # make sure that the new bounding box is not out of bounds
        scaled_new_x_min = min(max(scaled_new_x_min, 0), self.crop_width - 1)
        scaled_new_y_min = min(max(scaled_new_y_min, 0), self.crop_height - 1)
        scaled_new_x_max = min(max(scaled_new_x_max, 0), self.crop_width - 1)
        scaled_new_y_max = min(max(scaled_new_y_max, 0), self.crop_height - 1)

        return np.array([scaled_new_x_min, scaled_new_y_min, scaled_new_x_max, scaled_new_y_max])

    def __call__(self, tensor, bbox):

        H, W = tensor.shape[1], tensor.shape[2]
        
        bbox_width  = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        scale_factor_h = max(self.crop_height / self.max_upscale_factor, min(bbox_height+16, H)) / H
        scale_factor_w = max(self.crop_width / self.max_upscale_factor, min(bbox_width+16, W))  / W
        min_scale_factor = max(scale_factor_h, scale_factor_w)

        scale_factor = np.random.uniform(min_scale_factor, 1.0)

        if W > H:
            crop_height = int(H * scale_factor)
            crop_width  = int(crop_height / (self.crop_height / self.crop_width))
            if crop_width > W:
                crop_height = int(crop_height * (W / crop_width))
                crop_width  = W
        else:
            crop_width  = int(W * scale_factor)
            crop_height = int(crop_width / (self.crop_width / self.crop_height))
            if crop_height > H:
                crop_width  = int(crop_width * (H / crop_height))
                crop_height = H

        crop_coords = self.get_random_crop_coords(bbox, tensor.shape[1], tensor.shape[2], crop_height, crop_width)

        tensor = self.scale_crop(tensor, *crop_coords, crop_height, crop_width)
        bbox   = self.compute_new_bbox(bbox, crop_coords, crop_height, crop_width)

        return tensor, bbox

class ScaleCrop:
    def __init__(self, crop_size, max_upscale_factor=1.15):
        self.crop_height, self.crop_width = crop_size
        self.max_upscale_factor = max_upscale_factor

    def get_random_crop_coords(self, img_height, img_width, crop_height, crop_width):

        crop_y_max = max(img_height - crop_height, 0)
        crop_x_max = max(img_width - crop_width, 0)

        y1 = np.random.randint(0, crop_y_max) if crop_y_max > 0 else 0
        x1 = np.random.randint(0, crop_x_max) if crop_x_max > 0 else 0

        return y1, x1

    def scale_crop(self, tensor, y1, x1, crop_height, crop_width):
        cropped_tensor = tensor[:, :, y1:y1 + crop_height, x1:x1 + crop_width] if tensor.shape[2] >= crop_height and tensor.shape[3] >= crop_width else tensor
        if self.crop_height == crop_height and self.crop_width == crop_width:
            return cropped_tensor

        return F.interpolate(cropped_tensor, size=(self.crop_height, self.crop_width), mode='bilinear', align_corners=False)

    def __call__(self, tensor, seed = None):

        if seed is not None:
            np.random.seed(seed)

        H, W = tensor.shape[2], tensor.shape[3]

        scale_factor_h = self.crop_height / H
        scale_factor_w = self.crop_width  / W
        min_scale_factor = max(scale_factor_h, scale_factor_w) / self.max_upscale_factor

        scale_factor = np.random.uniform(min_scale_factor, 1)

        if W > H:
            crop_height = int(H * scale_factor)
            crop_width  = int(crop_height / (self.crop_height / self.crop_width))
            if crop_width > W:
                crop_height = int(crop_height * (W / crop_width))
                crop_width  = W
        else:
            crop_width  = int(W * scale_factor)
            crop_height = int(crop_width / (self.crop_width / self.crop_height))
            if crop_height > H:
                crop_width  = int(crop_width * (H / crop_height))
                crop_height = H

        crop_coords = self.get_random_crop_coords(tensor.shape[2], tensor.shape[3], crop_height, crop_width)

        return self.scale_crop(tensor, *crop_coords, crop_height, crop_width)
