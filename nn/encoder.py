import torch.nn as nn
import torch as th
import numpy as np
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, Prioritize, Binarize, MultiArgSequential
from nn.embedding import PositionPooling
from nn.hypernets import HyperSequential, HyperConvNextBlock, NonHyperWrapper
from nn.eprop_gate_l0rd import EpropGateL0rd
from torch.autograd import Function
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from nn.downscale import MemoryEfficientPatchDownScale

from typing import Tuple, Union, List
import utils
import cv2

__author__ = "Manuel Traub"

class NeighbourChannels(nn.Module):
    def __init__(self, channels):
        super(NeighbourChannels, self).__init__()

        self.register_buffer("weights", th.ones(channels, channels, 1, 1), persistent=False)

        for i in range(channels):
            self.weights[i,i,0,0] = 0

    def forward(self, input: th.Tensor):
        return nn.functional.conv2d(input, self.weights)

class InputPreprocessing(nn.Module):
    def __init__(self, num_slots: int, size: Union[int, Tuple[int, int]]): 
        super(InputPreprocessing, self).__init__()
        self.num_slots   = num_slots
        self.neighbours  = NeighbourChannels(num_slots)
        self.gaus2d      = Gaus2D(size)
        self.to_batch    = Rearrange('b (o c) -> (b o) c', o = num_slots)
        self.to_shared   = BatchToSharedObjects(num_slots)

        # backward compatibility
        self.register_buffer('legacy', th.tensor(0))
        self.max_std = 0.5
        self.min_std = 1/min(size)

    def forward(
        self, 
        input_rgb: th.Tensor, 
        input_depth: th.Tensor,
        error_last: th.Tensor, 
        mask: th.Tensor,
        mask_raw: th.Tensor,
        slot_rgb: th.Tensor,
        slot_depth: th.Tensor,
        slot_flow: th.Tensor,
        position: th.Tensor,
        slot_reset: th.Tensor,
    ):

        bg_mask     = repeat(mask[:,-1:], 'b 1 h w -> b c h w', c = self.num_slots)
        mask        = mask[:,:-1] * slot_reset.unsqueeze(-1).unsqueeze(-1)
        mask_others = self.neighbours(mask)
        position    = self.to_batch(position)

        # backward compatibility
        if self.legacy.item() == 1:
            std = position[:,-1:]
            std = th.sigmoid(std) * (self.max_std - self.min_std) + self.min_std
            position = th.cat((position[:,:-1], std), dim=1)

        own_gaus2d    = self.to_shared(self.gaus2d(position))

        input_rgb     = repeat(input_rgb,        'b c h w -> b o c h w', o = self.num_slots)
        input_depth   = repeat(input_depth,      'b c h w -> b o c h w', o = self.num_slots)
        error_last    = repeat(error_last,       'b 1 h w -> b o 1 h w', o = self.num_slots)
        bg_mask       = rearrange(bg_mask,       'b o h w -> b o 1 h w')
        mask_others   = rearrange(mask_others,   'b o h w -> b o 1 h w')
        mask          = rearrange(mask,          'b o h w -> b o 1 h w')
        slot_rgb      = rearrange(slot_rgb,    'b (o c) h w -> b o c h w', o = self.num_slots+1 if self.num_slots > 1 else 1)
        slot_depth    = rearrange(slot_depth,  'b (o c) h w -> b o c h w', o = self.num_slots)
        slot_flow     = rearrange(slot_flow,   'b (o c) h w -> b o c h w', o = self.num_slots)
        own_gaus2d    = rearrange(own_gaus2d,    'b o h w -> b o 1 h w')
        mask_raw      = rearrange(mask_raw,      'b o h w -> b o 1 h w')
        slot_reset    = rearrange(slot_reset,    'b o -> b o 1 1 1')

        output = th.cat((
            input_rgb, 
            input_depth, 
            error_last,
            mask, 
            mask_others,
            bg_mask,
            slot_rgb[:,:-1] * mask_raw * slot_reset if self.num_slots > 1 else slot_rgb * mask_raw * slot_reset,
            slot_depth * mask_raw * slot_reset,
            slot_flow * mask_raw * slot_reset,
            own_gaus2d,
            mask_raw * slot_reset,
        ), dim=2) 
        output = rearrange(output, 'b o c h w -> (b o) c h w')

        return output

class PatchDownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4):
        super(PatchDownConv, self).__init__()
        assert out_channels % in_channels == 0
        
        self.layers = nn.Linear(in_channels * kernel_size**2, out_channels)

        self.kernel_size     = kernel_size
        self.channels_factor = out_channels // in_channels

    def forward(self, input: th.Tensor):
        H, W = input.shape[2:]
        K    = self.kernel_size
        C    = self.channels_factor

        skip = reduce(input, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=K, w2=K)
        skip = repeat(skip, 'b c h w -> b (c n) h w', n=C)

        input    = rearrange(input, 'b c (h h2) (w w2) -> (b h w) (c h2 w2)', h2=K, w2=K)
        residual = self.layers(input)
        residual = rearrange(residual, '(b h w) c -> b c h w', h = H // K, w = W // K)

        return skip + residual

class PixelToPosition(nn.Module):
    def __init__(self, size): # TODO add update grid !!!
        super(PixelToPosition, self).__init__()

        self.register_buffer("grid_y", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_x", th.arange(size[1]), persistent=False)

        self.grid_y = (self.grid_y / (size[0]-1)) * 2 - 1
        self.grid_x = (self.grid_x / (size[1]-1)) * 2 - 1

        self.grid_y = self.grid_y.view(1, 1, -1, 1).expand(1, 1, *size).clone()
        self.grid_x = self.grid_x.view(1, 1, 1, -1).expand(1, 1, *size).clone()

        self.size = size

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1

        input = rearrange(input, 'b c h w -> b c (h w)')
        input = th.softmax(input, dim=2)
        input = rearrange(input, 'b c (h w) -> b c h w', h = self.size[0], w = self.size[1])

        x = th.sum(input * self.grid_x, dim=(2,3))
        y = th.sum(input * self.grid_y, dim=(2,3))

        return th.cat((x, y), dim=1)

class PixelToSTD(nn.Module):
    def __init__(self):
        super(PixelToSTD, self).__init__()

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1
        return reduce(input, 'b c h w -> b c', 'mean')

class PixelToDepth(nn.Module):
    def __init__(self):
        super(PixelToDepth, self).__init__()

    def forward(self, input):
        assert input.shape[1] == 1
        return th.sigmoid(reduce(input, 'b c h w -> b c', 'mean'))

class PixelToPriority(nn.Module):
    def __init__(self, num_slots):
        super(PixelToPriority, self).__init__()

        self.num_slots = num_slots
        self.register_buffer("indices", rearrange(th.arange(num_slots), 'a -> 1 a'), persistent=False)
        self.indices = (self.indices / (num_slots - 1)) * 2 - 1

        self.index_factor    = nn.Parameter(th.ones(1))
        self.priority_factor = nn.Parameter(th.zeros(1)+1e-16)
        self.depth_factor    = nn.Parameter(th.zeros(1)+1e-16)

    def forward(self, input: th.Tensor, depth: th.Tensor):
        assert input.shape[1] == 1
        priority = th.tanh(reduce(input, '(b o) 1 h w -> b o', 'mean', o = self.num_slots))
        priority = priority * self.priority_factor + self.index_factor * self.indices
        priority = rearrange(priority, 'b o -> (b o) 1') + depth * th.abs(self.depth_factor)
        return priority

class LociEncoder(nn.Module):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]], 
        latent_size: Union[int, Tuple[int, int]],
        num_slots: int, 
        base_channels: int,
        blocks,
        hyper_channels: int,
        gestalt_size: int,
        batch_size: int,
    ):
        super(LociEncoder, self).__init__()
        self.num_slots   = num_slots
        self.latent_size = latent_size

        img_channels = 16

        self.to_shared = Rearrange('(b o) c -> b (o c)', o = self.num_slots)
        self.to_batch  = Rearrange('b (o c) -> (b o) c', o = self.num_slots)

        self.preprocess = InputPreprocessing(num_slots, input_size)

        self.stem = HyperSequential(
            NonHyperWrapper(PatchDownConv(img_channels, base_channels)),
            *[HyperConvNextBlock(base_channels) for _ in range(blocks[0])],
            NonHyperWrapper(PatchDownConv(base_channels, base_channels * 2, kernel_size=2)),
            *[HyperConvNextBlock(base_channels * 2) for _ in range(blocks[1])],
            NonHyperWrapper(PatchDownConv(base_channels * 2, base_channels * 4, kernel_size=2)),
            *[HyperConvNextBlock(base_channels * 4) for _ in range(blocks[2])]
        )

        self.position_encoder = HyperSequential(
            *[HyperConvNextBlock(base_channels * 4) for _ in range(blocks[3])],
            HyperConvNextBlock(base_channels * 4, 4)
        )

        self.xy_encoder       = PixelToPosition(latent_size)
        self.std_encoder      = PixelToSTD()
        self.depth_encoder    = PixelToDepth()
        self.priority_encoder = PixelToPriority(num_slots)

        self.gestalt_base_encoder = HyperSequential(
            *[HyperConvNextBlock(base_channels * 4) for _ in range(blocks[3])],
        )

        self.mask_gestalt_encoder = HyperSequential(
            NonHyperWrapper(PatchDownConv(base_channels * 4, base_channels * 8, kernel_size=2)),
            *[HyperConvNextBlock(base_channels * 8) for _ in range(blocks[3] * 2)]
        )

        self.depth_gestalt_encoder = HyperSequential(
            NonHyperWrapper(PatchDownConv(base_channels * 4, base_channels * 8, kernel_size=2)),
            *[HyperConvNextBlock(base_channels * 8) for _ in range(blocks[3] * 2)]
        )

        self.rgb_gestalt_encoder = HyperSequential(
            NonHyperWrapper(PatchDownConv(base_channels * 4, base_channels * 8, kernel_size=2)),
            *[HyperConvNextBlock(base_channels * 8) for _ in range(blocks[3] * 2)]
        )

        self.mask_gestalt_pooling = MultiArgSequential(
            PositionPooling(
                in_channels  = base_channels * 8, 
                out_channels = gestalt_size,
                size         = [latent_size[0] // 2, latent_size[1] // 2]
            ),
            Binarize()
        )
        self.depth_gestalt_pooling = MultiArgSequential(
            PositionPooling(
                in_channels  = base_channels * 8, 
                out_channels = gestalt_size,
                size         = [latent_size[0] // 2, latent_size[1] // 2]
            ),
            Binarize()
        )
        self.depth_scale_pooling = MultiArgSequential(
            PositionPooling(
                in_channels  = base_channels * 8, 
                out_channels = 1,
                size         = [latent_size[0] // 2, latent_size[1] // 2]
            ),
            nn.Sigmoid()
        )
        self.rgb_gestalt_pooling = MultiArgSequential(
            PositionPooling(
                in_channels  = base_channels * 8, 
                out_channels = gestalt_size,
                size         = [latent_size[0] // 2, latent_size[1] // 2]
            ),
            Binarize()
        )

        self.hyper_weights = nn.Sequential(
            nn.Linear(gestalt_size * 3, hyper_channels),
            nn.SiLU(),
            *[nn.Sequential(
                nn.Linear(hyper_channels, hyper_channels),
                nn.SiLU()
            ) for _ in range(blocks[2]-2)],
            nn.Linear(hyper_channels, self.num_hyper_weights())
        )

    def num_hyper_weights(self):
        return (
            self.stem.num_weights()                  + 
            self.position_encoder.num_weights()      + 
            self.gestalt_base_encoder.num_weights()  +
            self.mask_gestalt_encoder.num_weights()  +
            self.depth_gestalt_encoder.num_weights() +
            self.rgb_gestalt_encoder.num_weights()
        )

    def forward(
        self, 
        input_rgb: th.Tensor,
        input_depth: th.Tensor,
        error_last: th.Tensor,
        mask: th.Tensor,
        mask_raw: th.Tensor,
        slot_rgb: th.Tensor,
        slot_depth: th.Tensor,
        slot_flow: th.Tensor,
        position: th.Tensor,
        gestalt: th.Tensor,
        slot_reset: th.Tensor = None,
        use_hyper_weights: bool = True,
    ):
        if slot_reset is None and use_hyper_weights:
            slot_reset = th.ones_like(position[:,:1])
        
        if slot_reset is None:
            slot_reset = th.zeros_like(position[:,:1])
        
        latent = self.preprocess(
            input_rgb, 
            input_depth, 
            error_last, 
            mask, 
            mask_raw,
            slot_rgb, 
            slot_depth, 
            slot_flow, 
            position,
            slot_reset
        )

        gestalt = self.to_batch(gestalt)[:,:-1] # remove z-scale
        slot_reset = self.to_batch(slot_reset)

        if use_hyper_weights:
            hyper_weights = self.hyper_weights(gestalt) * slot_reset

            stem_hyper_weights = hyper_weights[:,:self.stem.num_weights()]; 
            offset = self.stem.num_weights()

            position_hyper_weights = hyper_weights[:,offset:offset+self.position_encoder.num_weights()]; 
            offset += self.position_encoder.num_weights()

            base_gestalt_hyper_weights = hyper_weights[:,offset:offset+self.gestalt_base_encoder.num_weights()];
            offset += self.gestalt_base_encoder.num_weights()

            mask_gestalt_hyper_weights  = hyper_weights[:,offset:offset+self.mask_gestalt_encoder.num_weights()];
            offset += self.mask_gestalt_encoder.num_weights()

            depth_gestalt_hyper_weights = hyper_weights[:,offset:offset+self.depth_gestalt_encoder.num_weights()];
            offset += self.depth_gestalt_encoder.num_weights()

            rgb_gestalt_hyper_weights = hyper_weights[:,offset:offset+self.rgb_gestalt_encoder.num_weights()];
            offset += self.rgb_gestalt_encoder.num_weights()

        latent         = self.stem(latent,                           stem_hyper_weights           if use_hyper_weights else None)
        latent_gestalt = self.gestalt_base_encoder(latent,           base_gestalt_hyper_weights   if use_hyper_weights else None)
        mask_gestalt   = self.mask_gestalt_encoder(latent_gestalt,   mask_gestalt_hyper_weights   if use_hyper_weights else None)
        depth_gestalt  = self.depth_gestalt_encoder(latent_gestalt,  depth_gestalt_hyper_weights  if use_hyper_weights else None)
        rgb_gestalt    = self.rgb_gestalt_encoder(latent_gestalt, rgb_gestalt_hyper_weights if use_hyper_weights else None)

        latent   = self.position_encoder(latent, position_hyper_weights if use_hyper_weights else None)
        xy       = self.xy_encoder(latent[:,0:1])
        z        = self.depth_encoder(latent[:,1:2])
        std      = self.std_encoder(latent[:,2:3])
        priority = self.priority_encoder(latent[:,3:4], z)
        position = th.cat((xy, z, std), dim=1)

        mask_gestalt   = self.mask_gestalt_pooling(mask_gestalt, position)
        depth_scale    = self.depth_scale_pooling(depth_gestalt, position)
        depth_gestalt  = self.depth_gestalt_pooling(depth_gestalt, position)
        rgb_gestalt    = self.rgb_gestalt_pooling(rgb_gestalt, position)

        position = self.to_shared(position)
        gestalt  = self.to_shared(th.cat((mask_gestalt, depth_gestalt, rgb_gestalt, depth_scale), dim=1))
        priority = self.to_shared(priority)

        return position, gestalt, priority

