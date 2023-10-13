import torch.nn as nn
import torch as th
import numpy as np
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, LambdaModule, Binarize
from nn.eprop_gate_l0rd import EpropGateL0rd
from torch.autograd import Function
from einops import rearrange, repeat, reduce
from nn.eprop_gate_l0rd import ReTanh

from typing import Tuple, Union, List
import utils
import cv2

__author__ = "Manuel Traub"

class UpdateModule(nn.Module):
    def __init__(
        self,
        gestalt_size: int,
        num_hidden: list,
        num_layers: int,
        num_slots: int,
        gate_noise_level: float = 0.1,
        reg_lambda: float = 0.000005
    ):
        super(UpdateModule, self).__init__()

        num_inputs = 2 * (gestalt_size * 3 + 7)

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o=num_slots))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o=num_slots))

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            *[nn.Sequential(nn.SiLU(), nn.Linear(num_hidden, num_hidden)) for _ in range(num_layers-2)],
            nn.SiLU(),
            nn.Linear(num_hidden, 2),
            LambdaModule(lambda x: x + 3),
            ReTanh(reg_lambda, gate_noise_level),
        )

    def forward(
        self, 
        position_cur, 
        gestalt_cur, 
        priority_cur, 
        occlusion_cur,
        position_last, 
        gestalt_last, 
        priority_last, 
        occlusion_last
    ):
        position_cur   = self.to_batch(position_cur)
        gestalt_cur    = self.to_batch(gestalt_cur)
        priority_cur   = self.to_batch(priority_cur)
        occlusion_cur  = self.to_batch(occlusion_cur)
        position_last  = self.to_batch(position_last)
        gestalt_last   = self.to_batch(gestalt_last)
        priority_last  = self.to_batch(priority_last)
        occlusion_last = self.to_batch(occlusion_last)

        x = th.cat([
            position_cur,  gestalt_cur,  priority_cur,  occlusion_cur, 
            position_last, gestalt_last, priority_last, occlusion_last
        ], dim=-1)
        x = self.layers(x)

        position_gate = x[:, 0:1]
        gestalt_gate  = x[:, 1:2]

        position_cur = position_cur * position_gate + position_last * (1 - position_gate)
        gestalt_cur  = gestalt_cur  * gestalt_gate  + gestalt_last  * (1 - gestalt_gate)
        priority_cur = priority_cur * gestalt_gate  + priority_last * (1 - gestalt_gate)

        position_cur = self.to_shared(position_cur)
        gestalt_cur  = self.to_shared(gestalt_cur)
        priority_cur = self.to_shared(priority_cur)

        position_gate = self.to_shared(position_gate)
        gestalt_gate  = self.to_shared(gestalt_gate)

        return position_cur, gestalt_cur, priority_cur, position_gate, gestalt_gate

class AlphaAttention(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_slots,
        heads,
        dropout = 0.0
    ):
        super(AlphaAttention, self).__init__()

        self.to_sequence = LambdaModule(lambda x: rearrange(x, '(b o) c -> b o c', o = num_slots))
        self.to_batch    = LambdaModule(lambda x: rearrange(x, 'b o c -> (b o) c', o = num_slots))

        self.alpha     = nn.Parameter(th.zeros(1)+1e-12)
        self.attention = nn.MultiheadAttention(
            num_hidden, 
            heads, 
            dropout = dropout, 
            batch_first = True
        )

    def forward(self, x: th.Tensor):
        x = self.to_sequence(x)
        x = x + self.alpha * self.attention(x, x, x, need_weights=False)[0]
        return self.to_batch(x)

class EpropAlphaGateL0rd(nn.Module):
    def __init__(self, num_hidden, batch_size, reg_lambda):
        super(EpropAlphaGateL0rd, self).__init__()
        
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)
        self.l0rd  = EpropGateL0rd(
            num_inputs  = num_hidden, 
            num_hidden  = num_hidden, 
            num_outputs = num_hidden, 
            reg_lambda  = reg_lambda,
            batch_size = batch_size
        )

    def forward(self, input, hidden):
        output, hidden = self.l0rd(input, hidden)
        return input + self.alpha * output, hidden

class InputEmbeding(nn.Module):
    def __init__(self, num_inputs, num_hidden, heads):
        super(InputEmbeding, self).__init__()

        self.embeding = nn.Sequential(
            nn.Linear(num_inputs, num_hidden * heads),
            nn.SiLU(),
            nn.Linear(num_hidden * heads, num_inputs * heads),
            LambdaModule(lambda x: rearrange(x, 'b (n c) -> b n c', n = heads))
        )
        self.skip = LambdaModule(lambda x: repeat(x, 'b c -> b n c', n = heads))
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)

        self.exchange_gate = nn.Sequential(
            nn.Linear(num_hidden * heads, (num_hidden - num_inputs) * heads), 
            nn.Sigmoid(),
            LambdaModule(lambda x: rearrange(x, 'b (n c) -> b n c', n = heads))
        )
        self.exchange_code = nn.Sequential(
            nn.Linear(num_hidden * heads, (num_hidden - num_inputs) * heads), 
            nn.Tanh(),
            LambdaModule(lambda x: rearrange(x, 'b (n c) -> b n c', n = heads))
        )

    def forward(self, input: th.Tensor, hidden: th.Tensor):
        input  = self.skip(input) + self.alpha * self.embeding(input)
        hidden = self.exchange_gate(hidden) * self.exchange_code(hidden)

        output = th.cat([input, hidden], dim=2)
        return rearrange(output, 'b n c -> b (n c)')

class OutputEmbeding(nn.Module):
    def __init__(self, num_hidden, num_outputs, heads):
        super(OutputEmbeding, self).__init__()
        self.crop = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b (n c) -> b n c', n = heads)),
            LambdaModule(lambda x: x[:,:,0:num_outputs]),
            LambdaModule(lambda x: rearrange(x, 'b n c -> b (n c)')),
        )
        self.embeding = nn.Sequential(
            nn.Linear(num_outputs * heads, num_hidden * heads),
            nn.SiLU(),
            nn.Linear(num_hidden * heads, num_outputs),
        )
        self.skip = LambdaModule(
            lambda x: reduce(x, 'b (n c) -> b c', 'mean', n = heads)
        )
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)

    def forward(self, input: th.Tensor):
        input = self.crop(input)
        return self.skip(input) + self.alpha * self.embeding(input)

class EpropGateL0rdTransformer(nn.Module):
    def __init__(
        self, 
        num_inputs,
        num_hidden,
        num_layers,
        num_slots,
        batch_size,
        heads, 
        reg_lambda,
        dropout=0.0
    ):
        super(EpropGateL0rdTransformer, self).__init__()

        if num_inputs > num_hidden:
            raise ValueError('num_inputs must be less than or equal to num_hidden')

        self.register_buffer('hidden', th.zeros(batch_size * num_slots, num_hidden * heads), persistent=False)
        
        self.num_layers      = num_layers
        self.input_embeding  = InputEmbeding(num_inputs, num_hidden, heads)
        self.attention       = nn.Sequential(*[AlphaAttention(num_hidden * heads, num_slots, heads, dropout) for _ in range(num_layers)])
        self.l0rds           = nn.Sequential(*[EpropAlphaGateL0rd(num_hidden * heads, batch_size * num_slots, reg_lambda) for _ in range(num_layers+1)])
        self.output_embeding = OutputEmbeding(num_hidden, num_inputs, heads)

    def get_openings(self):
        openings = 0
        for i in range(self.num_layers+1):
            openings += self.l0rds[i].l0rd.openings.item()

        return openings / self.num_layers

    def get_hidden(self):
        return self.hidden

    def set_hidden(self, hidden):
        self.hidden = hidden

    def detach(self):
        self.hidden = self.hidden.detach()

    def reset_state(self):
        self.hidden = th.zeros_like(self.hidden)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x, self.hidden = self.l0rds[0](self.input_embeding(x, self.hidden), self.hidden)

        for i in range(self.num_layers):
            x, self.hidden = self.l0rds[i+1](self.attention(x), self.hidden)

        return self.output_embeding(x)

class InvertedBottleneck(nn.Module):
    def __init__(self, channels):
        super(InvertedBottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, channels*4),
            nn.SiLU(),
            nn.Linear(channels*4, channels),
        )

    def forward(self, x):
        return x + self.layers(x)

class LatentEpropPredictor(nn.Module): 
    def __init__(
        self, 
        heads: int, 
        layers: int,
        reg_lambda: float,
        num_slots: int, 
        num_hidden: int,
        gestalt_size: int, 
        batch_size: int,
    ):
        super(LatentEpropPredictor, self).__init__()
        self.num_slots = num_slots

        self.reg_lambda = reg_lambda
        self.predictor  = EpropGateL0rdTransformer(
            num_inputs  = gestalt_size * 3 + 6,
            num_hidden  = num_hidden,
            heads       = heads, 
            num_layers  = layers,
            num_slots   = num_slots,
            reg_lambda  = reg_lambda, 
            batch_size  = batch_size,
        )

        print("Predictor: Binary Bottleneck")
        self.gestalt_bottleneck = nn.Sequential(
            InvertedBottleneck(gestalt_size * 3 + 1),
            LambdaModule(lambda x: x - 5),
            Binarize(),
        )
                
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o=num_slots))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o=num_slots))

    def get_openings(self):
        return self.predictor.get_openings()

    def get_hidden(self):
        return self.predictor.get_hidden()

    def set_hidden(self, hidden):
        self.predictor.set_hidden(hidden)

    def forward(
        self, 
        position: th.Tensor, 
        gestalt: th.Tensor, 
        priority: th.Tensor,
    ):

        position = self.to_batch(position)
        gestalt  = self.to_batch(gestalt)
        priority = self.to_batch(priority)

        input  = th.cat((position, gestalt, priority), dim=1)
        output = self.predictor(input)

        position      = output[:,:4]
        delta_gestalt = self.gestalt_bottleneck(output[:,4:-1])
        priority      = output[:,-1:]
        
        # binary xor
        gestalt = gestalt + delta_gestalt - 2 * gestalt * delta_gestalt

        position = self.to_shared(position)
        gestalt  = self.to_shared(gestalt)
        priority = self.to_shared(priority)

        return position, gestalt, priority
