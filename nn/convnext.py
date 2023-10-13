import torch.nn as nn
import torch as th
import numpy as np
import nn as nn_modules
from einops import rearrange, repeat, reduce
from torch.autograd import Function
import torch.nn.functional as F
import math
from utils.utils import LambdaModule
from nn.downscale import MemoryEfficientPatchDownScale
from nn.upscale import MemoryEfficientUpscaling

from typing import Union, Tuple

__author__ = "Manuel Traub"

class SkipConnection(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale_factor: float = 1.0
        ):
        super(SkipConnection, self).__init__()
        assert scale_factor == 1 or int(scale_factor) > 1 or int(1 / scale_factor) > 1, f'invalid scale factor in SpikeFunction: {scale_factor}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

    def channel_skip(self, input: th.Tensor):
        in_channels  = self.in_channels
        out_channels = self.out_channels
        
        if in_channels == out_channels:
            return input

        if in_channels % out_channels == 0 or out_channels % in_channels == 0:

            if in_channels > out_channels:
                return reduce(input, 'b (c n) h w -> b c h w', 'mean', n = in_channels // out_channels)

            if out_channels > in_channels:
                return repeat(input, 'b c h w -> b (c n) h w', n = out_channels // in_channels)

        mean_channels = np.gcd(in_channels, out_channels)
        input = reduce(input, 'b (c n) h w -> b c h w', 'mean', n = in_channels // mean_channels)
        return repeat(input, 'b c h w -> b (c n) h w', n = out_channels // mean_channels)

    def scale_skip(self, input: th.Tensor):
        scale_factor = self.scale_factor

        if scale_factor == 1:
            return input

        if scale_factor > 1:
            return repeat(
                input, 
                'b c h w -> b c (h h2) (w w2)', 
                h2 = int(scale_factor),
                w2 = int(scale_factor)
            )

        height = input.shape[2]
        width  = input.shape[3]

        # scale factor < 1
        scale_factor = int(1 / scale_factor)

        if width % scale_factor == 0 and height % scale_factor == 0:
            return reduce(
                input, 
                'b c (h h2) (w w2) -> b c h w', 
                'mean', 
                h2 = scale_factor,
                w2 = scale_factor
            )

        if width >= scale_factor and height >= scale_factor:
            return nn.functional.avg_pool2d(
                input, 
                kernel_size = scale_factor,
                stride = scale_factor
            )

        assert width > 1 or height > 1
        return reduce(input, 'b c h w -> b c 1 1', 'mean')


    def forward(self, input: th.Tensor):

        if self.scale_factor > 1:
            return self.scale_skip(self.channel_skip(input))

        return self.channel_skip(self.scale_skip(input))

class LinearSkip(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(LinearSkip, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if num_inputs % num_outputs != 0 and num_outputs % num_inputs != 0:
            mean_channels = np.gcd(num_inputs, num_outputs)
            print(f"[WW] gcd skip: {num_inputs} -> {mean_channels} -> {num_outputs}")
            assert(False)

    def forward(self, input: th.Tensor):
        num_inputs  = self.num_inputs
        num_outputs = self.num_outputs
        
        if num_inputs == num_outputs:
            return input

        if num_inputs % num_outputs == 0 or num_outputs % num_inputs == 0:

            if num_inputs > num_outputs:
                return reduce(input, 'b (c n) -> b c', 'mean', n = num_inputs // num_outputs)

            if num_outputs > num_inputs:
                return repeat(input, 'b c -> b (c n)', n = num_outputs // num_inputs)

        mean_channels = np.gcd(num_inputs, num_outputs)
        input = reduce(input, 'b (c n) -> b c', 'mean', n = num_inputs // mean_channels)
        return repeat(input, 'b c -> b (c n)', n = num_outputs // mean_channels)


class MemoryEfficientBottleneckFunction(Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2):

        # reshape input tensor to 2D
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1
        
        # SiLU activation function using x * sigmoid(x)
        output2 = output1 * th.sigmoid(output1)

        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        
        # Save input tensor for backward pass
        ctx.save_for_backward(input, weight1, bias1, weight2)
        
        return output3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, bias1, weight2 = ctx.saved_tensors

        B, C, H, W = grad_output.shape
        grad_output = grad_output.permute(0, 2, 3, 1).reshape(B * H * W, -1)

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
        grad_input = th.matmul(grad_silu, weight1).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        grad_weight1 = th.matmul(grad_silu.t(), input)
        grad_bias1 = grad_silu.sum(dim=0)

        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2

class MemoryEfficientBottleneck(th.nn.Module):
    def __init__(self, in_features, out_features):
        super(MemoryEfficientBottleneck, self).__init__()
        hidden_features = max(out_features, in_features) * 4
        self.weight1 = th.nn.Parameter(th.randn(hidden_features, in_features))
        self.bias1   = th.nn.Parameter(th.zeros(hidden_features))
        self.weight2 = th.nn.Parameter(th.randn(out_features, hidden_features))
        self.bias2   = th.nn.Parameter(th.zeros(out_features))

        th.nn.init.xavier_uniform_(self.weight1)
        th.nn.init.xavier_uniform_(self.weight2)

    def forward(self, input):
        return MemoryEfficientBottleneckFunction.apply(input, self.weight1, self.bias1, self.weight2, self.bias2)

class ConvNeXtBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            channels_per_group = 32,
        ):
        super(ConvNeXtBlock, self).__init__()

        channels_per_group = min(channels_per_group, in_channels)

        if out_channels is None:
            out_channels = in_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.GroupNorm(in_channels // channels_per_group, in_channels),
            MemoryEfficientBottleneck(in_channels, out_channels),
        )

        self.skip  = SkipConnection(in_channels, out_channels)

    def forward(self, input: th.Tensor) -> th.Tensor:
        return self.layers(input) + self.skip(input)

class PatchUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4):
        super(PatchUpscale, self).__init__()
        assert in_channels % out_channels == 0
        
        self.skip = SkipConnection(in_channels, out_channels, scale_factor=kernel_size)

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = in_channels, 
                kernel_size  = 3,
                padding      = 1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels  = in_channels, 
                out_channels = out_channels, 
                kernel_size  = kernel_size,
                stride       = kernel_size,
            ),
        )

    def forward(self, input):
        return self.skip(input) + self.residual(input)

class ConvNeXtStem(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
        ):

        super(ConvNeXtStem, self).__init__()
        
        self.kernel_size = kernel_size
        self.layers = nn.Linear(in_channels * kernel_size**2, out_channels)

    def forward(self, input: th.Tensor) -> th.Tensor:
        K = self.kernel_size
        input = rearrange(input, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2 = K, w2 = K)
        return th.permute(self.layers(th.permute(input, [0, 2, 3, 1])), [0, 3, 1, 2])

class PatchDownscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4):
        super(PatchDownscale, self).__init__()
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

class ConvNeXtEncoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        base_channels, 
        blocks=[3,3,9,3], 
        return_features = False,
    ):
        super(ConvNeXtEncoder, self).__init__()
        self.return_features = return_features
        
        self.layer0 = nn.Sequential(
            PatchDownscale(in_channels, base_channels, kernel_size=4) if base_channels % in_channels == 0 else ConvNeXtStem(in_channels, base_channels),
            *[ConvNeXtBlock(base_channels) for _ in range(blocks[0])]
        )

        self.layer1 = nn.Sequential(
            PatchDownscale(base_channels, base_channels * 2, kernel_size=2) if blocks[1] > 0 else nn.Identity(),
            *[ConvNeXtBlock(base_channels * 2) for _ in range(blocks[1])]
        )

        self.layer2 = nn.Sequential(
            PatchDownscale(base_channels * 2, base_channels * 4, kernel_size=2) if blocks[2] > 0 else nn.Identity(),
            *[ConvNeXtBlock(base_channels * 4) for _ in range(blocks[2])]
        )

        self.layer3 = nn.Sequential(
            PatchDownscale(base_channels * 4, base_channels * 8, kernel_size=2) if blocks[3] > 0 else nn.Identity(),
            *[ConvNeXtBlock(base_channels * 8) for _ in range(blocks[3])]
        )
        

    def forward(self, input: th.Tensor):
        
        features  = [self.layer0(input)]
        features += [self.layer1(features[-1])]
        features += [self.layer2(features[-1])]
        features += [self.layer3(features[-1])]

        if self.return_features:
            return list(reversed(features))

        return features[-1]

class ConvNeXtDecoder(nn.Module):
    def __init__(
        self, 
        out_channels, 
        base_channels, 
        blocks=[3,3,9,3], 
    ):
        super(ConvNeXtDecoder, self).__init__()

        self.layer0 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 8) for _ in range(blocks[3])],
            PatchUpscale(base_channels * 8, base_channels * 4, kernel_size=2) if blocks[3] > 0 else nn.Identity(),
        )

        self.layer1 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 4) for _ in range(blocks[2])],
            PatchUpscale(base_channels * 4, base_channels * 2, kernel_size=2) if blocks[2] > 0 else nn.Identity(),
        )

        self.layer2 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 2) for _ in range(blocks[1])],
            PatchUpscale(base_channels * 2, base_channels, kernel_size=2) if blocks[1] > 0 else nn.Identity(),
        )

        self.layer3 = nn.Sequential(
            *[ConvNeXtBlock(base_channels) for _ in range(blocks[0])],
            PatchUpscale(base_channels, out_channels, kernel_size=4),
        )

    def forward(self, input: th.Tensor):
        
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

class ConvNeXtUnet(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels, 
        base_channels, 
        encoder_blocks=[3,3,9,3], 
        decoder_blocks=[3,3,9,3],
    ):
        super(ConvNeXtUnet, self).__init__()
        
        self.encoder = ConvNeXtEncoder(in_channels, base_channels, encoder_blocks, return_features=True)

        self.layer0 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 8) for _ in range(decoder_blocks[3])],
            PatchUpscale(base_channels * 8, base_channels * 4, kernel_size=2) if decoder_blocks[3] > 0 else nn.Identity(),
        )

        self.merge1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 4) for _ in range(decoder_blocks[1])],
            PatchUpscale(base_channels * 4, base_channels * 2, kernel_size=2),
        )

        self.merge2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 2) for _ in range(decoder_blocks[1])],
            PatchUpscale(base_channels * 2, base_channels, kernel_size=2),
        )

        self.merge3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.layer3 = nn.Sequential(
            *[ConvNeXtBlock(base_channels) for _ in range(decoder_blocks[1])],
            PatchUpscale(base_channels, out_channels, kernel_size=4),
        )

    def forward(self, input: th.Tensor):

        features = self.encoder(input)

        x = self.layer0(features[0])
        x = self.layer1(self.merge1(th.cat((x, features[1]), dim=1)))
        x = self.layer2(self.merge2(th.cat((x, features[2]), dim=1)))
        x = self.layer3(self.merge3(th.cat((x, features[3]), dim=1)))
        
        return x

class ConvNeXtPatchEmbeddingEncoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        base_channels, 
        blocks=[3,3,9,3], 
        return_features = False,
    ):
        super(ConvNeXtPatchEmbeddingEncoder, self).__init__()
        self.return_features = return_features
        
        self.layer0 = nn.Sequential(
            MemoryEfficientPatchDownScale(in_channels, base_channels, scale_factor=4),
            *[ConvNeXtBlock(base_channels) for _ in range(blocks[0])]
        )

        self.layer1 = nn.Sequential(
            MemoryEfficientPatchDownScale(base_channels, base_channels * 2, scale_factor=2) if blocks[1] > 0 else nn.Identity(),
            *[ConvNeXtBlock(base_channels * 2) for _ in range(blocks[1])]
        )

        self.layer2 = nn.Sequential(
            MemoryEfficientPatchDownScale(base_channels * 2, base_channels * 4, scale_factor=2) if blocks[2] > 0 else nn.Identity(),
            *[ConvNeXtBlock(base_channels * 4) for _ in range(blocks[2])]
        )

        self.layer3 = nn.Sequential(
            MemoryEfficientPatchDownScale(base_channels * 4, base_channels * 8, scale_factor=2) if blocks[3] > 0 else nn.Identity(),
            *[ConvNeXtBlock(base_channels * 8) for _ in range(blocks[3])]
        )
        

    def forward(self, input: th.Tensor):
        
        features  = [self.layer0(input)]
        features += [self.layer1(features[-1])]
        features += [self.layer2(features[-1])]
        features += [self.layer3(features[-1])]

        if self.return_features:
            return list(reversed(features))

        return features[-1]

class ConvNeXtPatchEmbeddingUnet(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels, 
        base_channels, 
        encoder_blocks=[3,3,9,3], 
        decoder_blocks=[3,3,9,3],
    ):
        super(ConvNeXtPatchEmbeddingUnet, self).__init__()
        
        self.encoder = ConvNeXtPatchEmbeddingEncoder(in_channels, base_channels, encoder_blocks, return_features=True)

        self.layer0 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 8) for _ in range(decoder_blocks[3])],
            MemoryEfficientUpscaling(base_channels * 8, base_channels * 4, scale_factor=2) if decoder_blocks[3] > 0 else nn.Identity(),
        )

        self.merge1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 4) for _ in range(decoder_blocks[1])],
            MemoryEfficientUpscaling(base_channels * 4, base_channels * 2, scale_factor=2),
        )

        self.merge2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 2) for _ in range(decoder_blocks[1])],
            MemoryEfficientUpscaling(base_channels * 2, base_channels, scale_factor=2),
        )

        self.merge3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.layer3 = nn.Sequential(
            *[ConvNeXtBlock(base_channels) for _ in range(decoder_blocks[1])],
            MemoryEfficientUpscaling(base_channels, out_channels, scale_factor=4),
        )

    def forward(self, input: th.Tensor):

        features = self.encoder(input)

        x = self.layer0(features[0])
        x = self.layer1(self.merge1(th.cat((x, features[1]), dim=1)))
        x = self.layer2(self.merge2(th.cat((x, features[2]), dim=1)))
        x = self.layer3(self.merge3(th.cat((x, features[3]), dim=1)))
        
        return x
