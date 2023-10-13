import torch as th
import torch.nn as nn
import numpy as np
from utils.utils import TanhAlpha, LambdaModule
from einops import rearrange, reduce, repeat

__author__ = "Manuel Traub"

class ObjectDiscovery(nn.Module):
    def __init__(
            self, 
            num_slots: int, 
            gestalt_size: int,
            object_permanence_strength: int,
            entity_pretraining_steps: int
        ):
        super(ObjectDiscovery, self).__init__()
        self.object_permanence_strength = object_permanence_strength
        self.gestalt_size = gestalt_size

        if object_permanence_strength < 0 or object_permanence_strength > 1:
            raise ValueError("object_permanence_strength must be in (0, 1)")
        
        if entity_pretraining_steps < 0:
            raise ValueError("entity_pretraining_steps must be > 0")

        if entity_pretraining_steps > 1e4:
            raise ValueError("entity_pretraining_steps must be < 1e4")

        self.num_slots = num_slots
        self.std       = nn.Parameter(th.zeros(1)-5)
        self.depth     = nn.Parameter(th.zeros(1)+5)

        self.init = TanhAlpha(start = -1e-4 * entity_pretraining_steps)
        self.register_buffer('priority', th.zeros(num_slots)-5, persistent=False)
        self.register_buffer('threshold', th.ones(1) * 0.8)
        self.last_mask = None

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_slots))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_slots))

    def load_pretrained(self, state):
        if "std" in state:
            self.std.data = state['std']

        if "depth" in state:
            self.depth.data = state['depth']

    def reset_state(self):
        self.last_mask = None

    def forward(
        self, 
        error: th.Tensor, 
        mask: th.Tensor = None, 
        position: th.Tensor = None,
        gestalt: th.Tensor = None,
        priority: th.Tensor = None
    ):

        batch_size = error.shape[0]
        device     = error.device

        object_permanence_strength = self.object_permanence_strength
        if self.init.get() < 1 and self.init.get() < object_permanence_strength:
            object_permanence_strength = self.init()

        if self.last_mask is None:
            self.last_mask = th.zeros((batch_size * self.num_slots, 1), device = device)

        if mask is not None:
            mask = reduce(mask[:,:-1], 'b c h w -> (b c) 1' , 'max').detach()

            if object_permanence_strength <= 0:
                self.last_mask = mask
            elif object_permanence_strength < 1:
                self.last_mask = th.maximum(self.last_mask, mask)
                self.last_mask = self.last_mask - th.relu(-1 * (mask - self.threshold) * (1 - object_permanence_strength))
            else:
                self.last_mask = th.maximum(self.last_mask, mask)

        mask = (self.last_mask > self.threshold).float().detach()

        gestalt_new  = th.zeros((batch_size * self.num_slots, self.gestalt_size), device = device)

        if gestalt is None:
            gestalt = gestalt_new
        else:
            gestalt = self.to_batch(gestalt) * mask + gestalt_new * (1 - mask)

        if priority is None:
            priority = repeat(self.priority, 'o -> (b o) 1', b = batch_size)
        else:
            priority = self.to_batch(priority) * mask + repeat(self.priority, 'o -> (b o) 1', b = batch_size) * (1 - mask)

        error_mask = (reduce(error, 'b c h w -> b c 1 1', 'max') > 0.1).float()
        error = error * error_mask + th.rand_like(error) * (1 - error_mask) 
        
        # Normalize error map to form a probability distribution and flatten it. 
        # Sample 'num_slots' number of indices from this distribution with replacement, 
        # and convert these indices into image x and y positions.
        error_map_normalized = error / th.sum(error, dim=(1,2,3), keepdim=True)
        error_map_flat = error_map_normalized.view(batch_size, -1)
        sampled_indices = th.multinomial(error_map_flat, num_samples=self.num_slots, replacement=True)
        y_positions = sampled_indices // error.shape[3]
        x_positions = sampled_indices % error.shape[3]

        # Convert positions from range [0, error.shape] to range [-1, 1]
        x_positions = x_positions.float() / (error.shape[3] / 2.0) - 1
        y_positions = y_positions.float() / (error.shape[2] / 2.0) - 1

        x_positions = self.to_batch(x_positions)
        y_positions = self.to_batch(y_positions)

        std = repeat(self.std, '1 -> (b o) 1', b = batch_size, o = self.num_slots)

        if position is None:
            z = repeat(self.depth, '1 -> (b o) 1', b = batch_size, o = self.num_slots)
            position = th.cat((x_positions, y_positions, z, std), dim=-1)
        else:
            z = repeat(self.depth, '1 -> (b o) 1', b = batch_size, o = self.num_slots)
            position = self.to_batch(position) * mask + th.cat((x_positions, y_positions, z, std), dim=1) * (1 - mask)

        return self.to_shared(position), self.to_shared(gestalt), self.to_shared(priority), self.to_shared(mask)
