import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nn.convnext import ConvNeXtBlock, PatchDownscale
from utils.utils import LambdaModule, MultiArgSequential, Binarize, RGB2YCbCr, YCbCr2RGB, Gaus2D, MaskCenter, Conv1x1
from nn.upscale import MemoryEfficientUpscaling
from nn.embedding import PositionEmbedding, PositionPooling
from nn.downscale import MemoryEfficientPatchDownScale
from einops import reduce

class MaskDepthToRGBDecoder(nn.Module):
    def __init__(
            self, 
            gestalt_size    = 256, 
            num_layers      = 5, 
            mask_channels   = 32,
            depth_channels  = 64, 
            rgb_channels    = 128,
            expand_ratio    = 4
        ):
        super(MaskDepthToRGBDecoder, self).__init__()
        
        self.mask_encoder = nn.Sequential(
            MemoryEfficientPatchDownScale(1, mask_channels, scale_factor = 16, expand_ratio = expand_ratio),
            nn.Tanh()
        )

        self.depth_encoder = nn.Sequential(
            MemoryEfficientPatchDownScale(1, depth_channels, scale_factor = 16, expand_ratio = expand_ratio),
            nn.Tanh()
        )

        self.position_embedding = PositionEmbedding(8)

        self.layers = nn.Sequential(
            Conv1x1(gestalt_size + depth_channels + mask_channels + 16, rgb_channels),
            *[ConvNeXtBlock(rgb_channels) for _ in range(num_layers)],
        )

        self.rgb_upscale = nn.Sequential(
            MemoryEfficientUpscaling(rgb_channels, 3, expand_ratio = expand_ratio, scale_factor = 16),
            nn.Sigmoid()
        )

        self.ycbcr_to_rgb = YCbCr2RGB()

    def load_embeddings(self, mask_encoder, depth_encoder, rgb_upscale):
        self.mask_encoder.load_state_dict(mask_encoder.state_dict())
        self.depth_encoder.load_state_dict(depth_encoder.state_dict())
        self.rgb_upscale.load_state_dict(rgb_upscale.state_dict())

    def forward(self, position, gestalt, mask, depth):
        latent_mask  = self.mask_encoder(mask)
        latent_depth = self.depth_encoder(depth * mask)
        max_mask     = reduce(mask, 'b c (h h2) (w w2) -> b c h w', 'max', h2=16, w2=16)
        gestalt      = gestalt.unsqueeze(-1).unsqueeze(-1) * max_mask

        self.position_embedding.update_grid(max_mask.shape[2:])
        embedding = self.position_embedding(position) * max_mask

        return self.rgb_upscale(self.layers(th.cat((gestalt, latent_mask, latent_depth, embedding), dim=1)))


class RGBPretrainer(nn.Module):
    def __init__(
        self, 
        size, 
        gestalt_size     = 256, 
        num_layers       = 5, 
        mask_channels    = 32,
        depth_channels   = 64, 
        rgb_channels     = 128,
        encoder_blocks   = [4,8],
        encoder_channels = 256,
        expand_ratio     = 4,
    ):
        super(RGBPretrainer, self).__init__()

        latent_size = [size[0] // 32, size[1] // 32]

        self.rgb_embedding = nn.Sequential(
            MemoryEfficientPatchDownScale(3, rgb_channels, expand_ratio = expand_ratio, scale_factor=16),
            nn.Tanh()
        )
        self.encoder = nn.Sequential(
            ConvNeXtBlock(rgb_channels, encoder_channels),
            *[ConvNeXtBlock(encoder_channels) for _ in range(encoder_blocks[0])],
            PatchDownscale(encoder_channels, encoder_channels*2, kernel_size=2),
            *[ConvNeXtBlock(encoder_channels*2, encoder_channels*2) for _ in range(encoder_blocks[1])],
            ConvNeXtBlock(encoder_channels*2, max(gestalt_size, encoder_channels*2)),
        )

        self.pool = MultiArgSequential(
            PositionPooling(latent_size, max(gestalt_size, encoder_channels*2), gestalt_size),
            Binarize()
        )

        self.mask_center = MaskCenter(size, combine=True)
        self.decoder = MaskDepthToRGBDecoder(
            gestalt_size    = gestalt_size,
            num_layers      = num_layers,
            mask_channels   = mask_channels,
            depth_channels  = depth_channels,
            rgb_channels    = rgb_channels,
            expand_ratio    = expand_ratio
        )

    def load_embeddings(self, mask_encoder, depth_encoder, rgb_embedding, rgb_upscale):
        self.decoder.load_embeddings(mask_encoder, depth_encoder, rgb_upscale)
        self.rgb_embedding.load_state_dict(rgb_embedding.state_dict())

    def encode(self, mask, depth, rgb):
        position = self.mask_center(mask)
        gestalt  = self.pool(self.encoder(self.rgb_embedding(rgb * mask)), position)
        return gestalt

    def decode(self, position, gestalt, mask, depth):
        return self.decoder(position, gestalt, mask, depth)

    def forward(self, mask, depth, rgb):
        position  = self.mask_center(mask)
        gestalt   = self.pool(self.encoder(self.rgb_embedding(rgb * mask)), position)

        return self.decoder(position, gestalt, mask, depth), gestalt, position

