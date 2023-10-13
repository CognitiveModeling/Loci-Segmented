import torch.nn as nn
import torch as th
import numpy as np
from nn.convnext import SkipConnection
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, Prioritize, LambdaModule, MultiArgSequential
from torch.autograd import Function
from einops import rearrange, repeat, reduce
import torch.nn.functional as F

from typing import Tuple, Union, List
import utils

__author__ = "Manuel Traub"

class GestaltPositionMerge(nn.Module):
    def __init__(
        self, 
        latent_size: Union[int, Tuple[int, int]],
        num_slots: int,
    ):

        super(GestaltPositionMerge, self).__init__()
        self.num_slots = num_slots

        self.gaus2d = Gaus2D(size=latent_size)

        self.to_batch  = SharedObjectsToBatch(num_slots)
        self.to_shared = BatchToSharedObjects(num_slots)

        self.prioritize = Prioritize(num_slots)

    def forward(self, position, gestalt, priority = None, compute_raw = False):
        
        gestalt  = rearrange(gestalt, 'b c -> b c 1 1')
        position = self.gaus2d(position)

        if priority is not None:
            if compute_raw:
                gestalt  = th.cat((gestalt, gestalt))
                position = th.cat((
                    position,
                    self.to_batch(self.prioritize(self.to_shared(position), priority))
                ))
            else:
                position = self.to_batch(self.prioritize(self.to_shared(position), priority))

        return position * gestalt

class LociDecoder(nn.Module):
    def __init__(
        self, 
        latent_size: Union[int, Tuple[int, int]],
        gestalt_size: int,
        num_slots: int, 
        mask_decoder,
        depth_decoder,
        rgb_decoder,
    ): 

        super(LociDecoder, self).__init__()
        self.to_batch     = SharedObjectsToBatch(num_slots)
        self.to_shared    = BatchToSharedObjects(num_slots)
        self.num_slots    = num_slots
        self.latent_size  = latent_size
        self.gestalt_size = gestalt_size

        self.merge = GestaltPositionMerge(latent_size, num_slots)
        self.to_mask  = mask_decoder
        self.to_depth = depth_decoder
        self.to_rgb   = rgb_decoder

    def forward(self, position, gestalt, priority = None, compute_raw = False, gt_depth = None, gt_mask = None):

        position = rearrange(position, 'b (o c) -> (b o) c', o = self.num_slots)
        gestalt  = rearrange(gestalt, 'b (o c) -> (b o) c', o = self.num_slots)
        z_scale  = rearrange(gestalt[:,-1:], 'b c -> b c 1 1')
        z        = rearrange(position[:,2:3], 'b c -> b c 1 1')
        mask_gestalt, depth_gestalt, rgb_gestalt = th.chunk(gestalt[:,:-1], 3, dim=1)

        mask = self.to_mask(self.merge(position, mask_gestalt, priority, compute_raw))

        mask_raw = mask
        if compute_raw and priority is not None:
            mask_raw, mask = th.chunk(mask,2)

        mask_in = th.softmax(th.cat((mask_raw, th.ones_like(mask_raw)), dim=1), dim=1)[:,:1] if gt_mask is None else gt_mask

        depth_raw = self.to_depth(position, depth_gestalt, mask_in)
        depth     = z + depth_raw * z_scale

        depth_in = depth_raw if gt_depth is None else gt_depth

        rgb = self.to_rgb(position, rgb_gestalt, mask_in, depth_in)

        mask      = self.to_shared(mask)
        mask_raw  = self.to_shared(mask_raw)
        depth     = self.to_shared(depth)
        depth_raw = self.to_shared(depth_raw)
        rgb       = self.to_shared(rgb)

        return mask, rgb, depth, mask_raw, depth_raw
