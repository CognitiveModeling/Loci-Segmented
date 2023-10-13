import torch.nn as nn
import torch as th
import numpy as np
from torch.autograd import Function
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Resize, InterpolationMode

from typing import Tuple, Union, List
import utils

__author__ = "Manuel Traub"

class TanhAlpha(nn.Module):
    def __init__(self, start = 0, stepsize = 1e-4, max_value = 1):
        super(TanhAlpha, self).__init__()

        self.register_buffer('init', th.zeros(1) + start)
        self.stepsize  = stepsize
        self.max_value = max_value

    def get(self):
        return (th.tanh(self.init) * self.max_value).item()

    def forward(self):
        self.init = self.init.detach() + self.stepsize
        return self.get()

class MultiArgSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(MultiArgSequential, self).__init__(*args, **kwargs)

    def forward(self, *tensor):

        for n in range(len(self)):
            if isinstance(tensor, th.Tensor) or tensor == None:
                tensor = self[n](tensor)
            else:
                tensor = self[n](*tensor)

        return tensor

class Gaus2D(nn.Module):
    def __init__(self, size = None, position_limit = 1):
        super(Gaus2D, self).__init__()
        self.size = size
        self.position_limit = position_limit
        self.min_std = 0.1
        self.max_std = 0.5

        self.register_buffer("grid_x", th.zeros(1,1,1,1), persistent=False)
        self.register_buffer("grid_y", th.zeros(1,1,1,1), persistent=False)

        if size is not None:
            self.min_std = 1.0 / min(size)
            self.update_grid(size)

        print(f"Gaus2D: min std: {self.min_std}")

    def update_grid(self, size):

        if size != self.grid_x.shape[2:]:
            self.size    = size
            self.min_std = 1.0 / min(size)
            H, W = size

            self.grid_x = th.arange(W, device=self.grid_x.device)
            self.grid_y = th.arange(H, device=self.grid_x.device)

            self.grid_x = (self.grid_x / (W-1)) * 2 - 1
            self.grid_y = (self.grid_y / (H-1)) * 2 - 1

            self.grid_x = self.grid_x.view(1, 1, 1, -1).expand(1, 1, H, W).clone()
            self.grid_y = self.grid_y.view(1, 1, -1, 1).expand(1, 1, H, W).clone()

    def forward(self, input: th.Tensor):
        assert input.shape[1] >= 2 and input.shape[1] <= 4
        H, W = self.size

        x   = rearrange(input[:,0:1], 'b c -> b c 1 1')
        y   = rearrange(input[:,1:2], 'b c -> b c 1 1')
        std = th.zeros_like(x)

        if input.shape[1] == 3:
            std = rearrange(input[:,2:3], 'b c -> b c 1 1')

        if input.shape[1] == 4:
            std = rearrange(input[:,3:4], 'b c -> b c 1 1')

        x   = th.clip(x, -self.position_limit, self.position_limit)
        y   = th.clip(y, -self.position_limit, self.position_limit)
        std = th.clip(std, self.min_std, self.max_std)
            
        std_y = std.clone()
        std_x = std * (H / W)

        return th.exp(-1 * ((self.grid_x - x)**2/(2 * std_x**2) + (self.grid_y - y)**2/(2 * std_y**2)))

class SharedObjectsToBatch(nn.Module):
    def __init__(self, num_slots):
        super(SharedObjectsToBatch, self).__init__()

        self.num_slots = num_slots

    def forward(self, input: th.Tensor):
        return rearrange(input, 'b (o c) h w -> (b o) c h w', o=self.num_slots)

class BatchToSharedObjects(nn.Module):
    def __init__(self, num_slots):
        super(BatchToSharedObjects, self).__init__()

        self.num_slots = num_slots

    def forward(self, input: th.Tensor):
        return rearrange(input, '(b o) c h w -> b (o c) h w', o=self.num_slots)

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, *x):
        return self.lambd(*x)

class Prioritize(nn.Module):
    def __init__(self, num_slots):
        super(Prioritize, self).__init__()

        self.num_slots = num_slots

    def forward(self, input: th.Tensor, priority: th.Tensor):
        
        if priority is None:
            return input

        priority = priority * 250 + th.randn_like(priority) * 2.5

        batch_size = input.shape[0]
        weights    = th.zeros((batch_size, self.num_slots, self.num_slots, 1, 1), device=input.device)

        for o in range(self.num_slots):
            weights[:,o,:,0,0] = th.sigmoid(priority[:,:] - priority[:,o:o+1])
            weights[:,o,o,0,0] = weights[:,o,o,0,0] * 0

        input  = rearrange(input, 'b c h w -> 1 (b c) h w')
        weights = rearrange(weights, 'b o i 1 1 -> (b o) i 1 1')

        output = th.relu(input - nn.functional.conv2d(input, weights, groups=batch_size))
        output = rearrange(output, '1 (b c) h w -> b c h w ', b=batch_size)

        return output

class Binarize(nn.Module):
    def __init__(self):
        super(Binarize, self).__init__()

    def forward(self, input: th.Tensor):
        input = th.sigmoid(input)
        if not self.training:
            return th.round(input)

        return input + input * (1 - input) * th.randn_like(input)

class MaskCenter(nn.Module):
    def __init__(self, size, normalize=True, combine=False):
        super(MaskCenter, self).__init__()
        self.combine = combine

        # Get the mask dimensions
        height, width = size

        # Create meshgrid of coordinates
        if normalize:
            x_range = th.linspace(-1, 1, width)
            y_range = th.linspace(-1, 1, height)
        else:
            x_range = th.linspace(0, width, width)
            y_range = th.linspace(0, height, height)

        y_coords, x_coords = th.meshgrid(y_range, x_range)

        # Broadcast the coordinates to match the mask shape
        self.register_buffer('x_coords', x_coords[None, None, :, :], persistent=False)
        self.register_buffer('y_coords', y_coords[None, None, :, :], persistent=False)

    def forward(self, mask):

        # Compute the center of the mask for each instance in the batch
        center_x = th.sum(self.x_coords * mask, dim=(2, 3)) / (th.sum(mask, dim=(2, 3)) + 1e-8)
        center_y = th.sum(self.y_coords * mask, dim=(2, 3)) / (th.sum(mask, dim=(2, 3)) + 1e-8)
        std      = (th.sum(mask, dim=(2, 3)) / th.sum(th.ones_like(mask), dim=(2, 3)))**0.5

        if self.combine:
            return th.cat((center_x, center_y, std), dim=-1)

        return th.cat((center_x, center_y), dim=-1), std

class PositionInMask(nn.Module):
    """
    Computes a random position that lies inside the mask
    """
    def __init__(self, size):
        super(PositionInMask, self).__init__()

        # Get the mask dimensions
        height, width = size

        # Create meshgrid of coordinates
        x_range = th.linspace(-1, 1, width)
        y_range = th.linspace(-1, 1, height)

        y_coords, x_coords = th.meshgrid(y_range, x_range)

        # Broadcast the coordinates to match the mask shape
        self.register_buffer('x_coords', x_coords[None, None, :, :], persistent=False)
        self.register_buffer('y_coords', y_coords[None, None, :, :], persistent=False)

    def forward(self, mask):

        B, C, H, W = mask.shape

        with th.no_grad():
            bin_mask     = (mask > 0.75).float()
            erroded_mask = 1 - th.nn.functional.max_pool2d(1 - bin_mask, kernel_size=5, stride=1, padding=2)

            use_center = (th.sum(erroded_mask, dim=(2, 3)) < 0.1).float()

            rand_mask = th.randn_like(erroded_mask) * erroded_mask * 1000
            rand_pixel = th.softmax(rand_mask.view(B, C, -1), dim=-1).view(B, C, H, W) * erroded_mask

            # Compute the center of the mask for each instance in the batch
            center_x = th.sum(self.x_coords * mask, dim=(2, 3)) / (th.sum(mask, dim=(2, 3)) + 1e-6)
            center_y = th.sum(self.y_coords * mask, dim=(2, 3)) / (th.sum(mask, dim=(2, 3)) + 1e-6)
            std      = (th.sum(mask, dim=(2, 3)) / th.sum(th.ones_like(mask), dim=(2, 3)))**0.5

            # compute the random position inside the mask for each instance in the batch
            rand_x = th.sum(self.x_coords * rand_pixel, dim=(2, 3))
            rand_y = th.sum(self.y_coords * rand_pixel, dim=(2, 3))
            
            center_pos = th.cat((center_x, center_y), dim=-1)
            rand_pos   = th.cat((rand_x, rand_y), dim=-1)

            return use_center * center_pos + (1 - use_center) * rand_pos, center_pos, std

        assert False, "This should never happen"
        return None

# RGB to YCbCr
class RGB2YCbCr(nn.Module):
    def __init__(self):
        super(RGB2YCbCr, self).__init__()

        kr = 0.299
        kg = 0.587
        kb = 0.114

        # The transformation matrix from RGB to YCbCr (ITU-R BT.601 conversion)
        self.register_buffer("matrix", th.tensor([
            [                  kr,                  kg,                    kb],
            [-0.5 * kr / (1 - kb), -0.5 * kg / (1 - kb),                  0.5],
            [                 0.5, -0.5 * kg / (1 - kr), -0.5 * kb / (1 - kr)]
        ]).t(), persistent=False)

        # Adjustments for each channel
        self.register_buffer("shift", th.tensor([0., 0.5, 0.5]), persistent=False)

    def forward(self, img):
        if len(img.shape) != 4 or img.shape[1] != 3:
            raise ValueError('Input image must be 4D tensor with a size of 3 in the second dimension.')

        return th.tensordot(img.permute(0, 2, 3, 1), self.matrix, dims=1).permute(0, 3, 1, 2) + self.shift[None, :, None, None]

# RGBD to YCbCr
class RGBD2YCbCr(nn.Module):
    def __init__(self):
        super(RGBD2YCbCr, self).__init__()

        kr = 0.299
        kg = 0.587
        kb = 0.114

        # The transformation matrix from RGB to YCbCr (ITU-R BT.601 conversion)
        self.register_buffer("matrix", th.tensor([
            [                  kr,                  kg,                    kb],
            [-0.5 * kr / (1 - kb), -0.5 * kg / (1 - kb),                  0.5],
            [                 0.5, -0.5 * kg / (1 - kr), -0.5 * kb / (1 - kr)]
        ]).t(), persistent=False)

        # Adjustments for each channel
        self.register_buffer("shift", th.tensor([0., 0.5, 0.5]), persistent=False)

    def forward(self, img):
        if len(img.shape) != 4 or img.shape[1] != 4:
            raise ValueError('Input image must be 4D tensor with a size of 4 in the second dimension.')

        ycbcr = th.tensordot(img[:,:3].permute(0, 2, 3, 1), self.matrix, dims=1).permute(0, 3, 1, 2) + self.shift[None, :, None, None]
        return th.cat((ycbcr, img[:,3:]), dim=1)


# YCbCr to RGB
class YCbCr2RGB(nn.Module):
    def __init__(self):
        super(YCbCr2RGB, self).__init__()

        kr = 0.299
        kg = 0.587
        kb = 0.114

        # The transformation matrix from YCbCr to RGB (ITU-R BT.601 conversion)
        self.register_buffer("matrix", th.tensor([
            [1,                       0,              2 - 2 * kr],
            [1, -kb / kg * (2 - 2 * kb), -kr / kg * (2 - 2 * kr)],
            [1,              2 - 2 * kb,                       0]
        ]).t(), persistent=False)

        # Adjustments for each channel
        self.register_buffer("shift", th.tensor([0., 0.5, 0.5]), persistent=False)

    def forward(self, img):
        if len(img.shape) != 4 or img.shape[1] != 3:
            raise ValueError('Input image must be 4D tensor with a size of 3 in the second dimension.')

        result = th.tensordot((img - self.shift[None, :, None, None]).permute(0, 2, 3, 1), self.matrix, dims=1).permute(0, 3, 1, 2)

        # Clamp the results to the valid range for RGB [0, 1]
        return th.clamp(result, 0, 1)

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.in_channels)
        x = self.linear(x)
        x = x.view(batch_size, height, width, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class GradientScaler(nn.Module):
    class ScaleGrad(Function):
        @staticmethod
        def forward(ctx, input_tensor, scale):
            ctx.scale = scale
            return input_tensor.clone()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * ctx.scale, None

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        return self.ScaleGrad.apply(input, self.scale)

class RadomSimilarityBasedMaskDrop(nn.Module):
    def __init__(self, sigma_scale = 25):
        super(RadomSimilarityBasedMaskDrop, self).__init__()
        self.sigma_scale = sigma_scale

        
    def distance_weights(self, positions):
        sigma = positions[:,:,-1]
        positions = positions[:,:,:-1]

        # give more weight to z distance
        positions = th.cat((positions, positions[:,:,2:3]), dim=2)
        
        # Expand dims to compute pairwise differences
        p1 = positions[:, :, None, :]
        p2 = positions[:, None, :, :]

        # expand sigma
        sigma1 = sigma[:,:,None]
        sigma2 = sigma[:,None,:]
        
        # Compute pairwise differences and squared Euclidean distance
        diff = p1 - p2
        squared_diff = diff ** 2
        squared_distances = th.sum(squared_diff, dim=-1)

        var = sigma1 * sigma2 * self.sigma_scale
        
        # Compute the actual distances
        distances = th.sqrt(squared_distances)
        weights = th.exp(-distances / (2 * var + 1e-5))
        
        return weights

    def batch_covariance(self, slots):
        mean_slots = th.mean(slots, dim=1, keepdim=True)
        centered_slots = slots - mean_slots
        cov_matrix = th.bmm(centered_slots.transpose(1, 2), centered_slots) / (slots.size(1) - 1)
        return cov_matrix

    def batch_correlation(self, slots):
        cov_matrix = self.batch_covariance(slots)
        variances = th.diagonal(cov_matrix, dim1=-2, dim2=-1)
        std_matrix = th.sqrt(variances[:, :, None] * variances[:, None, :])
        corr_matrix = cov_matrix / std_matrix
        return corr_matrix

    def get_drop_mask(self, similarity_matrix):
        similarity_matrix = th.relu(similarity_matrix)
        mean_similarity   = th.mean(similarity_matrix)
        similarity_matrix = th.relu(similarity_matrix - mean_similarity) / (1 - mean_similarity)
        similarity_matrix = th.triu(similarity_matrix) * (1 - th.eye(similarity_matrix.shape[-1], device=similarity_matrix.device))
        drop_propability  = reduce(similarity_matrix, 'b n m -> b n', 'max')
        return (drop_propability < 0.00001).float()

    def forward(self, position, gestalt, mask):
        num_slots = mask.shape[1]

        gestalt  = rearrange(gestalt,  'b (o c) -> b c o', o = num_slots)
        position = rearrange(position, 'b (o c) -> b c o', o = num_slots)

        visible  = th.softmax(th.cat((mask, th.ones_like(mask[:,:1])), dim=1), dim=1) 
        visible  = (reduce(visible[:,:-1], 'b c h w -> b 1 c', 'max') > 0.75).float().detach()
        gestalt  = gestalt.detach()  * visible  + (1 - visible) * (0.5 + th.randn_like(gestalt)*0.01)
        position = position.detach() * visible

        weights      = self.distance_weights(rearrange(position, 'b c o -> b o c'))
        slot_corr    = self.batch_correlation(gestalt) * weights 
        drop_mask    = self.get_drop_mask(slot_corr).unsqueeze(-1).unsqueeze(-1).detach()

        return mask * drop_mask - 10 * (1 - drop_mask)
