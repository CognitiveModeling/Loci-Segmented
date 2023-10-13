import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nn.convnext import ConvNeXtBlock, PatchDownscale
from utils.utils import LambdaModule, MultiArgSequential, Binarize, Gaus2D, MaskCenter
from nn.upscale import MemoryEfficientUpscaling
from nn.downscale import MemoryEfficientPatchDownScale
from nn.embedding import PositionPooling
from einops import rearrange, repeat, reduce

class MaskDecoder(nn.Module):
    def __init__(self, gestalt_size = 256, mask_channels = 32, expand_ratio = 4):
        super(MaskDecoder, self).__init__()
        assert gestalt_size // 2 >= mask_channels

        decoder = [nn.Conv2d(gestalt_size, gestalt_size // 2, kernel_size = 3, padding = 1)]

        for i in range(1,10):
            if gestalt_size // 2**(i + 1) <= mask_channels:
                decoder.append(nn.SiLU())
                decoder.append(nn.Conv2d(gestalt_size // 2**i, mask_channels, kernel_size = 3, padding = 1))
                break

            decoder.append(nn.SiLU())
            decoder.append(nn.Conv2d(gestalt_size // 2**i, gestalt_size // 2**(i + 1), kernel_size = 3, padding = 1))

        self.decoder = nn.Sequential(*decoder)
        self.mask_reconstruction = MemoryEfficientUpscaling(mask_channels, 1, scale_factor=16, expand_ratio=expand_ratio)

    def load_mask_reconstruction(self, mask_reconstruction):
        self.mask_reconstruction.load_state_dict(mask_reconstruction.state_dict())

    def forward(self, x):
        x = self.decoder(x)
        return self.mask_reconstruction(x)
                
class MaskPretrainer(nn.Module):
    def __init__(
        self, 
        size, 
        gestalt_size     = 256, 
        mask_channels    = 32, 
        encoder_blocks   = [2,4], 
        encoder_channels = 256,
        expand_ratio     = 4,
    ):
        super(MaskPretrainer, self).__init__()

        self.mask_embedding = nn.Sequential(
            MemoryEfficientPatchDownScale(1, mask_channels, scale_factor=16, expand_ratio=expand_ratio),
            nn.Tanh()
        )
        self.encoder = nn.Sequential(
            ConvNeXtBlock(mask_channels, encoder_channels),
            *[ConvNeXtBlock(encoder_channels, encoder_channels) for _ in range(encoder_blocks[0])],
            PatchDownscale(encoder_channels, encoder_channels*2, kernel_size=2),
            *[ConvNeXtBlock(encoder_channels*2, encoder_channels*2) for _ in range(encoder_blocks[1])],
            ConvNeXtBlock(encoder_channels*2, max(gestalt_size, encoder_channels*2)),
        )
        self.pool = MultiArgSequential(
            PositionPooling([size[0] // 32, size[1] // 32], max(gestalt_size, encoder_channels*2), gestalt_size),
            Binarize(),
        )
        self.gaus2d      = Gaus2D([size[0] // 16, size[1] // 16])
        self.mask_center = MaskCenter(size, combine=True)
            
        self.decoder = MaskDecoder(gestalt_size, mask_channels, expand_ratio)

    def load_embeddings(self, mask_embedding, mask_reconstruction):
        self.mask_embedding.load_state_dict(mask_embedding.state_dict())
        self.decoder.load_mask_reconstruction(mask_reconstruction)

    def encode(self, x):
        position = self.mask_center(x)
        gestalt  = self.pool(self.encoder(self.mask_embedding(x)), position).unsqueeze(-1).unsqueeze(-1)
        return gestalt

    def decode(self, gestalt, position):
        position2d = self.gaus2d(position)
        x = self.decoder(gestalt * position2d)
        return x

    def forward(self, x):
        position  = self.mask_center(x)
        gestalt   = self.pool(self.encoder(self.mask_embedding(x)), position).unsqueeze(-1).unsqueeze(-1)
    
        position2d = self.gaus2d(position)

        x = self.decoder(gestalt * position2d)

        return x, gestalt, position

