import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, repeat, reduce
from typing import Tuple, Union, List

class PatchDownScaleFunction(Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2, scale_factor, residual):
        
        # Reshape input tensor to 2D
        B, C, H, W = input.shape
        permuted_input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        permuted_input = permuted_input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # First linear layer
        output1 = th.matmul(permuted_input, weight1.t()) + bias1
        
        # SiLU activation function
        output2 = output1 * th.sigmoid(output1)
        
        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight1, bias1, weight2, bias2)
        ctx.scale_factor = scale_factor
        ctx.residual = residual

        if residual:
            input = reduce(input, 'b c (h s1) (w s2) -> b c h w', 'mean', s1=scale_factor, s2=scale_factor)
            input = repeat(input, 'b c h w -> b (c n) h w', n=output3.shape[1] // C)
            output3 = output3 + input
        
        return output3
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, bias1, weight2, bias2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        
        # Recompute input
        B, C, H, W = input.shape
        permuted_input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        permuted_input = permuted_input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # Recompute necessary outputs for backward pass
        output1 = th.matmul(permuted_input, weight1.t()) + bias1
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for second linear layer
        grad_output2 = grad_output.permute(0, 2, 3, 1).reshape(B * H // scale_factor * W // scale_factor, -1)
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2   = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)
        
        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)
        
        # Gradients for first linear layer
        grad_weight1 = th.matmul(grad_silu.t(), permuted_input)
        grad_bias1   = grad_silu.sum(dim=0)
        
        # Gradients for gestalt and embedding
        #grad_input = th.matmul(grad_silu, weight1).reshape(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        grad_input = th.matmul(grad_silu, weight1)
        grad_input = grad_input.reshape(B, H // scale_factor, W // scale_factor, C, scale_factor, scale_factor)
        grad_input = grad_input.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if ctx.residual:
            grad_output = reduce(grad_output, 'b (c n) h w -> b c h w', 'sum', n=grad_output.shape[1] // C)
            grad_output = repeat(grad_output, 'b c h w -> b c (h s1) (w s2)', s1=scale_factor, s2=scale_factor) / (scale_factor ** 2)
            grad_input  = grad_input + grad_output
        
        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None, None


class MemoryEfficientPatchDownScale(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4, scale_factor=2, residual=False):
        super(MemoryEfficientPatchDownScale, self).__init__()
        
        self.scale_factor = scale_factor
        self.residual = residual

        hidden_channels = max(in_channels, out_channels) * expand_ratio
        
        self.weight1 = nn.Parameter(th.randn(hidden_channels, in_channels * scale_factor * scale_factor))
        self.bias1   = nn.Parameter(th.zeros(hidden_channels))
        
        self.weight2 = nn.Parameter(th.randn(out_channels, hidden_channels))
        self.bias2   = nn.Parameter(th.zeros(out_channels))

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, input):
        return PatchDownScaleFunction.apply(
            input, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor, self.residual
        )
