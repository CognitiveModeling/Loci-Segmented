import torch.nn as nn
import torch as th
import numpy as np
import torch.nn.functional as F
from nn.vit import AttentionLayer, AttentionSum, MemoryEfficientBottleneck, CrossAttentionLayer, UncertaintyMaskedPatchEmbedding
from utils.utils import LambdaModule, Binarize, MultiArgSequential, GradientScaler, RGBD2YCbCr, RGB2YCbCr
from nn.hypernets import HyperSequential
from nn.convnext import ConvNeXtBlock, ConvNeXtUnet
from nn.hypernets import HyperConvNextBlock, HyperSequential, NonHyperWrapper
from nn.upscale import MemoryEfficientUpscaling
from nn.downscale import MemoryEfficientPatchDownScale
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import cv2
from utils.io import Timer

from typing import Union, Tuple

__author__ = "Manuel Traub"

class ContextEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, binarize = True):
        super(ContextEncoding, self).__init__()

        self.skip      = LambdaModule(lambda x: repeat(x, 'b n c -> b n (c c2)', c2 = out_channels // in_channels))
        self.expand    = MemoryEfficientBottleneck(in_channels, out_channels)
        self.aggregate = AttentionSum(out_channels)
        self.process   = nn.Sequential(nn.Linear(out_channels, out_channels*4), nn.SiLU(), nn.Linear(out_channels*4, out_channels))
        self.binarize  = Binarize() if binarize else nn.Identity()

    def forward(self, x):
        x = self.skip(x) + self.expand(x)
        x = self.aggregate(x)
        x = self.binarize(x + self.process(x))
        return x

class AffineGridGenerator(nn.Module):
    def __init__(self):
        super(AffineGridGenerator, self).__init__()

    def forward(self, params, size):
        # Construct the identity transformation matrix
        theta = th.zeros(params.shape[0], 2, 3, device=params.device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1

        # Add the residuals from the input tensor to the identity matrix
        theta += params.view(-1, 2, 3)

        # Generate and return the affine grid
        return rearrange(F.affine_grid(theta, size), 'b h w c -> (b h w) c')

class MotionContextEncoder(nn.Module):
    def __init__(
        self, 
        context_size, 
        in_channels, 
        latent_channels, 
        num_layers, 
        uncertainty_threshold,
    ):
        super(MotionContextEncoder, self).__init__()
        self.uncertainty_threshold = uncertainty_threshold
        self.context_size = context_size
        self.latent_channels = latent_channels

        self.to_patches = nn.Sequential(
            MemoryEfficientPatchDownScale(in_channels, latent_channels, scale_factor = 16, expand_ratio = 4),
            Rearrange('b c h w -> b (h w) c')
        )

        embedd_hidden = 2 * latent_channels
        self.source_embedding = nn.Sequential(
            nn.Linear(2, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, latent_channels)
        )
        self.target_embedding = nn.Sequential(
            nn.Linear(2, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, latent_channels)
        )

        self.layers = nn.Sequential(
            *[AttentionLayer(latent_channels) for _ in range(num_layers)],
            ContextEncoding(latent_channels, context_size, binarize = False),
            nn.Linear(context_size, 6),
        )

        self.grid_generator = AffineGridGenerator()

        self.motion_embedding = nn.Sequential(
            nn.Linear(2, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, embedd_hidden),
            nn.SiLU(),
            nn.Linear(embedd_hidden, context_size)
        )

    def compute_embedding(self, B, H, W, device):
        grid_y, grid_x = th.meshgrid(
            th.linspace(-1, 1, H, device = device),
            th.linspace(-1, 1, W, device = device),
            indexing='ij'
        )

        grid_x = grid_x.reshape(1, 1, H, W).clone()
        grid_y = grid_y.reshape(1, 1, H, W).clone()

        grid = rearrange(th.cat((grid_x, grid_y), dim=1), '1 c h w -> (h w) c')

        source_embedding = self.source_embedding(grid)
        target_embedding = self.target_embedding(grid)

        source_embedding = repeat(source_embedding, 'n c -> b n c', b = B)
        target_embedding = repeat(target_embedding, 'n c -> b n c', b = B)

        return source_embedding, target_embedding

    def forward(self, source, target, source_uncertainty, target_uncertainty, delta_t):
        B, _, H, W = source.shape
        latent_size = (B, self.latent_channels, H // 16, W // 16)

        source_embedding, target_embedding = self.compute_embedding(B, H // 16, W // 16, source.device)

        if th.max(th.abs(delta_t)) < 0.5:
            motion_context   = th.zeros((B, 6), device = source.device)
            motion_embedding = self.motion_embedding(self.grid_generator(motion_context, latent_size))
            motion_embedding = rearrange(motion_embedding, '(b h w) c -> b c h w', b = B, h = H // 16, w = W // 16)
            return motion_embedding, motion_context

        t_mask = (th.abs(delta_t) > 0.5).float()

        source_uncertainty = (source_uncertainty < self.uncertainty_threshold).float()
        target_uncertainty = (target_uncertainty < self.uncertainty_threshold).float()

        source = self.to_patches(source * source_uncertainty) + source_embedding
        target = self.to_patches(target * target_uncertainty) + target_embedding

        motion_context = self.layers(th.cat((source, target), dim = 1)) * t_mask

        motion_embedding = self.motion_embedding(self.grid_generator(motion_context, latent_size))
        motion_embedding = rearrange(motion_embedding, '(b h w) c -> b c h w', b = B, h = H // 16, w = W // 16)

        return motion_embedding, motion_context

class DepthContextDecoder(nn.Module):
    def __init__(self, depth_context_size, motion_context_size, channels, num_layers):
        super(DepthContextDecoder, self).__init__()

        self.layers     = nn.Sequential(
            ConvNeXtBlock(depth_context_size, channels),
            *[ConvNeXtBlock(channels) for _ in range(num_layers-1)]
        )
        self.to_patches = MemoryEfficientUpscaling(channels, 1, scale_factor = 16)

    def forward(self, depth_context, motion_embedding):
        latent = depth_context.unsqueeze(-1).unsqueeze(-1) + motion_embedding
        return self.to_patches(self.layers(latent))

class RGBContextDecoder(nn.Module):
    def __init__(self, context_size, channels, cross_attention_layers, convnext_layers):
        super(RGBContextDecoder, self).__init__()

        self.cross_attention_layers = nn.Sequential(*[CrossAttentionLayer(channels) for _ in range(cross_attention_layers)])
        self.convnext_layers        = nn.Sequential(*[ConvNeXtBlock(channels) for _ in range(convnext_layers)])
        self.to_patches             = nn.Sequential(MemoryEfficientUpscaling(channels, 3, scale_factor = 16), nn.Sigmoid())
        self.depth_embedding        = nn.Sequential(GradientScaler(0.1), MemoryEfficientPatchDownScale(1, context_size, scale_factor = 16))
        self.preprocess             = nn.Sequential(
            ConvNeXtBlock(context_size, channels),
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, rgb_patches, depth, motion_embedding):
        
        x = self.preprocess(self.depth_embedding(depth) + motion_embedding)
            
        for layer in self.cross_attention_layers:
            x = layer(x, rgb_patches)

        x = rearrange(x, 'b (h w) c -> b c h w', h = motion_embedding.shape[-2], w = motion_embedding.shape[-1])

        return self.to_patches(self.convnext_layers(x))

class UncertantyBackground(nn.Module):
    def __init__(
        self, 
        uncertainty_threshold,
        motion_context_size,
        depth_context_size,
        latent_channels,
        num_layers,
        depth_input = False,
    ):
        super(UncertantyBackground, self).__init__()

        self.base_encoder = MultiArgSequential(
            UncertaintyMaskedPatchEmbedding(
                input_channels          = 4 if depth_input else 3,
                latent_channels         = latent_channels,
                uncertainty_threshold   = uncertainty_threshold,
            ),
            *[AttentionLayer(latent_channels) for _ in range(num_layers)]
        )

        self.rgb_encoder   = nn.Sequential(
            *[AttentionLayer(latent_channels) for _ in range(num_layers//2)],
        )
        self.depth_encoder = nn.Sequential(
            *[AttentionLayer(latent_channels) for _ in range(num_layers//2)],
            ContextEncoding(latent_channels, depth_context_size),
        )

        self.motion_encoder = MotionContextEncoder(
            context_size            = motion_context_size,
            in_channels             = 4 if depth_input else 3,
            latent_channels         = latent_channels,
            num_layers              = num_layers + num_layers // 2,
            uncertainty_threshold   = uncertainty_threshold,
        )

        self.depth_decoder = DepthContextDecoder(
            depth_context_size  = depth_context_size,
            motion_context_size = motion_context_size,
            channels            = latent_channels,
            num_layers          = num_layers,
        )

        self.rgb_decoder = RGBContextDecoder(
            context_size           = motion_context_size,
            channels               = latent_channels,
            cross_attention_layers = num_layers,
            convnext_layers        = num_layers // 2,
        )

        self.uncertainty_estimation = nn.Sequential(
            RGBD2YCbCr() if depth_input else RGB2YCbCr(),
            ConvNeXtUnet(
                in_channels    = 4 if depth_input else 3,
                out_channels   = 1,
                base_channels  = latent_channels // 4,
                encoder_blocks = [max(1, num_layers // 4), max(1, num_layers // 2), max(1, num_layers), max(1, num_layers // 2)],
                decoder_blocks = [max(1, num_layers // 8), max(1, num_layers // 4), max(1, num_layers), max(1, num_layers // 4)],
            ),
            nn.Sigmoid(),
            LambdaModule(lambda x: (x, x + x * (1 - x) * th.randn_like(x))),
        )

        self.to_ycbcr = RGBD2YCbCr() if depth_input else RGB2YCbCr()

    def get_last_layer(self):
        return None

    def forward(self, source, target, source_uncertainty, target_uncertainty, delta_t, color_input):

        source = self.to_ycbcr(source)
        target = self.to_ycbcr(target)
        
        source_y, source_cbcr, source_depth = source[:, 0:1], source[:, 1:3], source[:, 3:4]
        target_y, target_cbcr, target_depth = target[:, 0:1], target[:, 1:3], target[:, 3:4]

        source = th.cat([source_y, target_cbcr * color_input, source_depth], dim = 1)
        target = th.cat([target_y, target_cbcr * color_input, target_depth], dim = 1)

        latent = self.base_encoder(source, source_uncertainty)
        rgb_patches, depth_context = self.rgb_encoder(latent), self.depth_encoder(latent)

        motion_embedding, motion_context = self.motion_encoder(source, target, source_uncertainty, target_uncertainty, delta_t)

        depth = self.depth_decoder(depth_context, motion_embedding)
        rgb   = self.rgb_decoder(rgb_patches, depth, motion_embedding)

        return rgb, depth, motion_context, depth_context

