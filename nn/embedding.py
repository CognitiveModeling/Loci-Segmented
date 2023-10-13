import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import RGB2YCbCr, YCbCr2RGB, Gaus2D
from nn.convnext import LinearSkip
from nn.upscale import MemoryEfficientUpscaling
from nn.downscale import MemoryEfficientPatchDownScale
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat, reduce  

class PositionEmbedding(nn.Module):
    def __init__(self, num_frequencies, max_std = 0.5, position_limit = 1.0):
        super(PositionEmbedding, self).__init__()

        self.register_buffer('grid_x', th.zeros(0), persistent=False)
        self.register_buffer('grid_y', th.zeros(0), persistent=False)

        self.num_frequencies = num_frequencies
        self.max_std = max_std
        self.position_limit = position_limit
        
        # backward compatibility
        self.register_buffer('legacy', th.tensor(0))

    def update_grid(self, size):
        if self.grid_x is None or self.grid_x.shape[2:] != size:

            self.size = size
            H, W = size

            self.min_std = 1/min(size)

            grid_x = th.linspace(-1, 1, W, device=self.grid_x.device)
            grid_y = th.linspace(-1, 1, H, device=self.grid_y.device)

            grid_y, grid_x = th.meshgrid(grid_y, grid_x, indexing='ij')

            self.grid_x = grid_x.reshape(1, 1, H, W).clone()
            self.grid_y = grid_y.reshape(1, 1, H, W).clone()

    def forward(self, input: th.Tensor):
        assert input.shape[1] >= 2 and input.shape[1] <= 4

        x   = rearrange(input[:,0:1], 'b c -> b c 1 1')
        y   = rearrange(input[:,1:2], 'b c -> b c 1 1')
        std = th.zeros_like(x)

        if input.shape[1] == 3:
            std = rearrange(input[:,2:3], 'b c -> b c 1 1')

        if input.shape[1] == 4:
            std = rearrange(input[:,3:4], 'b c -> b c 1 1')
        
        # backward compatibility
        if self.legacy.item() == 1:
            std = th.sigmoid(std) * (self.max_std - self.min_std) + self.min_std

        x   = th.clip(x, -self.position_limit, self.position_limit)
        y   = th.clip(y, -self.position_limit, self.position_limit)
        std = 0.1 / th.clip(std, self.min_std, self.max_std)

        H, W = self.size
        std_y = std.clone()
        std_x = std * (W/H)

        grid_x = (self.grid_x - x) * std_x * np.pi/2
        grid_y = (self.grid_y - y) * std_y * np.pi/2

        embedding = []

        for i in range(self.num_frequencies):
            embedding.append(th.cos(grid_x * 2**i))
            embedding.append(th.cos(grid_y * 2**i))

        return th.cat(embedding, dim=1)

class PositionPooling(nn.Module):
    def __init__(self, size, in_channels, out_channels):
        super(PositionPooling, self).__init__()
        self.gaus2d = Gaus2D(size)
        self.skip = LinearSkip(in_channels, out_channels)
        self.residual = nn.Sequential(
            nn.Linear(in_channels, max(in_channels, out_channels) * 4),
            nn.SiLU(),
            nn.Linear(max(in_channels, out_channels) * 4, out_channels)
        )

        # backward compatibility
        self.register_buffer('legacy', th.tensor(0))
        self.max_std = 0.5
        self.min_std = 1/min(size)
        
    def forward(self, feature_maps, position):

        # backward compatibility
        if self.legacy.item() == 1:
            std = position[:,-1:]
            std = th.sigmoid(std) * (self.max_std - self.min_std) + self.min_std
            position = th.cat((position[:,:-1], std), dim=1)

        mask = self.gaus2d(position)
        mask = mask / (reduce(mask, 'b c h w -> b 1 1 1', 'sum') + 1e-8)

        x = reduce(mask * feature_maps, 'b c h w -> b c', 'sum')
        return self.skip(x) + self.residual(x)

