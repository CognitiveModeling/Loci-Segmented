import torch.nn as nn
import torch as th
import numpy as np
from nn.eprop_gate_l0rd import EpropGateL0rd
from utils.utils import LambdaModule
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from typing import Tuple, Union, List
from torch.autograd import Function
from nn.downscale import MemoryEfficientPatchDownScale
from nn.upscale import MemoryEfficientUpscaling

__author__ = "Manuel Traub"

class UncertaintyMaskedPatchEmbedding(nn.Module):
    def __init__(
        self, 
        input_channels,
        latent_channels,
        expand_ratio = 4,
        uncertainty_threshold = 0.1,
        embedding = True
    ):
        super(UncertaintyMaskedPatchEmbedding, self).__init__()

        self.uncertainty_threshold = uncertainty_threshold

        if embedding:
            embedd_hidden = 2 * latent_channels
            self.embedding = nn.Sequential(
                nn.Linear(2, embedd_hidden),
                nn.SiLU(),
                nn.Linear(embedd_hidden, embedd_hidden),
                nn.SiLU(),
                nn.Linear(embedd_hidden, embedd_hidden),
                nn.SiLU(),
                nn.Linear(embedd_hidden, latent_channels),
            )
        else:
            self.embedding = False

        self.to_patches = nn.Sequential(
            MemoryEfficientPatchDownScale(input_channels, latent_channels, scale_factor = 16, expand_ratio = expand_ratio),
            Rearrange('b c h w -> b (h w) c')
        )

    def compute_embedding(self, B, H, W, device):

        grid_y, grid_x = th.meshgrid(
            th.linspace(-1, 1, H, device=device), 
            th.linspace(-1, 1, W, device=device),
            indexing='ij'
        )

        grid_x = grid_x.reshape(1, 1, H, W).clone()
        grid_y = grid_y.reshape(1, 1, H, W).clone()

        grid = rearrange(th.cat((grid_x, grid_y), dim=1), '1 c h w -> (h w) c')

        return repeat(self.embedding(grid), 'n c -> b n c', b=B)

    def forward(self, input, uncertainty, embedding = None):
        B, _, H, W = input.shape

        uncertainty = reduce(uncertainty, 'b 1 (h h2) (w w2) -> b (h w) 1', 'max', h2=16, w2=16)
        patch_mask  = (uncertainty < self.uncertainty_threshold).float()

        latent = self.to_patches(input)

        if embedding is None and self.embedding:
            embedding = self.compute_embedding(input.shape[0], H // 16, W // 16, input.device)

        return (latent + embedding) * patch_mask

class MemoryEfficientBottleneckFunction(Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2):

        # reshape input tensor to 2D
        B, N, C = input.shape
        input = input.reshape(B * N, -1)

        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1
        
        # SiLU activation function using x * sigmoid(x)
        output2 = output1 * th.sigmoid(output1)

        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        
        # Save input tensor for backward pass
        ctx.save_for_backward(input, weight1, bias1, weight2)
        
        return output3.reshape(B, N, -1)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, bias1, weight2 = ctx.saved_tensors

        B, N, C = grad_output.shape
        grad_output = grad_output.reshape(B * N, -1)

        # Recalculate necessary outputs for backward pass
        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1
        
        # SiLU activation function using x * sigmoid(x)
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for second linear layer
        grad_output2 = grad_output
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2 = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)

        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)

        # Gradients for first linear layer
        grad_input = th.matmul(grad_silu, weight1).reshape(B, N, -1)
        grad_weight1 = th.matmul(grad_silu.t(), input)
        grad_bias1 = grad_silu.sum(dim=0)

        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2

class MemoryEfficientBottleneck(th.nn.Module):
    def __init__(self, in_features, out_features):
        super(MemoryEfficientBottleneck, self).__init__()
        self.weight1 = th.nn.Parameter(th.randn(out_features * 4, in_features))
        self.bias1   = th.nn.Parameter(th.zeros(out_features * 4))
        self.weight2 = th.nn.Parameter(th.randn(out_features, out_features * 4))
        self.bias2   = th.nn.Parameter(th.zeros(out_features))

        th.nn.init.xavier_uniform_(self.weight1)
        th.nn.init.xavier_uniform_(self.weight2)

    def forward(self, input):
        return MemoryEfficientBottleneckFunction.apply(input, self.weight1, self.bias1, self.weight2, self.bias2)

class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        head_size = 64,
        dropout = 0.0
    ):
        super(AttentionLayer, self).__init__()

        self.norm1 = nn.LayerNorm(num_hidden)
        self.attention = nn.MultiheadAttention(
            num_hidden, 
            min(1, num_hidden // head_size), 
            dropout = dropout, 
            batch_first = True
        )
        self.norm2 = nn.LayerNorm(num_hidden)
        self.mlp   = MemoryEfficientBottleneck(num_hidden, num_hidden)

    def forward(self, x: th.Tensor):
        norm_x = self.norm1(x)
        x = x + self.attention(norm_x, norm_x, norm_x, need_weights=False)[0]

        return x + self.mlp(self.norm2(x))


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        head_size = 64,
        dropout = 0.0
    ):
        super(CrossAttentionLayer, self).__init__()

        self.norm1 = nn.LayerNorm(num_hidden)
        self.cross_attention = nn.MultiheadAttention(
            num_hidden, 
            min(1, num_hidden // head_size), 
            dropout = dropout, 
            batch_first = True
        )
        self.norm2 = nn.LayerNorm(num_hidden)
        self.mlp   = MemoryEfficientBottleneck(num_hidden, num_hidden)

    def forward(self, x: th.Tensor, context: th.Tensor):
        norm_x = self.norm1(x)
        x = x + self.cross_attention(norm_x, context, context, need_weights=False)[0]

        return x + self.mlp(self.norm2(x))

class AttentionSum(nn.Module):
    def __init__(
        self,
        num_hidden,
        head_size = 64,
        dropout = 0.0
    ):
        super(AttentionSum, self).__init__()

        self.query = nn.Parameter(th.randn(1, 1, num_hidden))
        self.norm1 = nn.LayerNorm(num_hidden)
        self.alpha = nn.Parameter(th.zeros(1))

        self.attention = nn.MultiheadAttention(
            num_hidden, 
            min(1, num_hidden // head_size), 
            dropout = dropout, 
            batch_first = True
        )

    def forward(self, x: th.Tensor):
        norm_x   = self.norm1(x)
        query    = repeat(self.query, '1 1 c -> b 1 c', b = x.shape[0])
        skip     = reduce(x, 'b s c -> b c', 'mean')
        residual = rearrange(self.attention(query, norm_x, norm_x, need_weights=False)[0], 'b 1 c -> b c')
        alpha    = th.sigmoid(self.alpha)

        return (skip * alpha + residual * (1 - alpha)).squeeze(1)
