import torch.nn as nn
import torch as th
import numpy as np
from torch.autograd import Function
import utils
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from nn.convnext import MemoryEfficientBottleneck, SkipConnection

from typing import Union, Tuple

__author__ = "Manuel Traub"

class HyperSequential(nn.Module):
    def __init__(self, *hyper_modules, use_base_weights = True):
        super(HyperSequential, self).__init__()
        self.use_base_weights = use_base_weights

        self.layers = nn.Sequential(*hyper_modules)

        if use_base_weights:
            self.base_weights  = nn.Parameter(th.zeros(1, self.num_weights()))
            self.alpha         = nn.Parameter(th.zeros(1) + 1e-16)

            self.ini_weights_(self.base_weights.data)

    def __getitem__(self, index):
        return self.layers[index]

    def ini_weights_(self, weights):
        
        offset = 0
        for layer in self.layers:
            layer.ini_weights_(weights[:,offset:offset+layer.num_weights()])
            offset += layer.num_weights()

    def num_weights(self):
        num_weights = 0
        for layer in self.layers:
            num_weights += layer.num_weights()

        return num_weights

    def forward(self, x, dynamic_weights):
        
        offset = 0
        for layer in self.layers:
            if self.use_base_weights:
                hyper_weights = self.base_weights[:,offset:offset+layer.num_weights()] 
                hyper_weights = hyper_weights.expand(x.shape[0], -1)
                if dynamic_weights is not None:
                    hyper_weights = hyper_weights + self.alpha * dynamic_weights[:,offset:offset+layer.num_weights()]
                    
                x = layer(x, hyper_weights)
            else:
                x = layer(x, dynamic_weights[:,offset:offset+layer.num_weights()])

        return x

class NonHyperWrapper(nn.Module):
    def __init__(self, base_layer):
        super(NonHyperWrapper, self).__init__()
        self.base_layer = base_layer

    def num_weights(self):
        return 0

    def ini_weights_(self, weights):
        pass

    def forward(self, x, dynamic_weights=None):
        return self.base_layer(x)


class HyperConv2dDepthWise(nn.Module):
    def __init__(
        self, 
        mlp_channels,
        mlp_kernel,
        act = None
    ):
        super(HyperConv2dDepthWise, self).__init__()
        self.mlp_channels = mlp_channels
        self.mlp_kernel   = mlp_kernel
        self.act          = act
        
        self.num_mlp_weights = mlp_kernel**2 * mlp_channels + mlp_channels

        print(f'HyperConv2dDepthWise: {mlp_kernel}^2 x {mlp_channels}')

    def num_weights(self):
        return self.num_mlp_weights

    def ini_weights_(self, weights):

        mlp_channels = self.mlp_channels
        mlp_kernel   = self.mlp_kernel

        # slice and rearrange hidden weights and biases
        offset = mlp_kernel**2 * mlp_channels
        w = weights[:,:offset]
        b = weights[:,offset:offset + mlp_channels]

        weights[:,:offset]                      = th.randn_like(w) * np.sqrt(6 / (mlp_kernel**2 + mlp_channels))
        weights[:,offset:offset + mlp_channels] = th.randn_like(b) * np.sqrt(1 / mlp_channels)

    def forward(self, input, weights):

        batch_size   = input.shape[0]
        mlp_channels = self.mlp_channels
        mlp_kernel   = self.mlp_kernel

        output = rearrange(input, 'b c h w -> 1 (b c) h w', c = mlp_channels)
        
        # slice and rearrange hidden weights and biases
        offset = mlp_kernel**2 * mlp_channels
        w = weights[:,:offset]
        b = weights[:,offset:offset + mlp_channels]
        w = rearrange(w, 'b (o h w) -> (b o) 1 h w', b = batch_size, o = mlp_channels, h = mlp_kernel, w = mlp_kernel)
        b = rearrange(b, 'b c -> (b c)')

        output = F.conv2d(output, w, bias=b, groups=batch_size*mlp_channels, padding='same')

        if self.act is not None:
            output = self.act(output)

        return rearrange(output, '1 (b c) h w -> b c h w', c = mlp_channels) 

class HyperConvNextBlock(nn.Module):
    def __init__(
        self, 
        mlp_input,
        mlp_output = None,
        skip = None,
        alpha = None
    ):
        super(HyperConvNextBlock, self).__init__()

        if mlp_output is None:
            mlp_output = mlp_input

        group_size = min(32, mlp_input)
        
        self.hyper_conv2d = HyperConv2dDepthWise(mlp_input, 7)
        self.linear_layers = nn.Sequential(
            nn.GroupNorm(mlp_input // group_size, mlp_input),
            MemoryEfficientBottleneck(mlp_input, mlp_output),
        )

        self.skip = SkipConnection(mlp_input, mlp_output) if skip is None else skip

        if alpha is not None:
            self.alpha = nn.Parameter(th.zeros(1) + alpha)
        else:
            self.alpha = None

        print(f'PartialHyperConvNextBlock: {mlp_input} -> {mlp_output}')

    def num_weights(self):
        return self.hyper_conv2d.num_weights()

    def ini_weights_(self, weights):
        self.hyper_conv2d.ini_weights_(weights)

    def forward(self, x, weights):
        if self.alpha is not None:
            return self.skip(x) + self.alpha * self.linear_layers(self.hyper_conv2d(x, weights))
        return self.skip(x) + self.linear_layers(self.hyper_conv2d(x, weights))

