import torch as th
import torch.nn as nn
from nn.convnext import SkipConnection
from scipy.optimize import linear_sum_assignment
from nn.upscale import MemoryEfficientUpscaling
from nn.convnext import ConvNeXtBlock, ConvNeXtEncoder
from nn.embedding import PositionPooling
from nn.mask_decoder import MaskDecoder
from utils.utils import LambdaModule, MultiArgSequential, Gaus2D, MaskCenter, Binarize

class UncertaintyPertrainer(nn.Module):
    def __init__(
            self, 
            size, 
            in_channels, 
            base_channels   = 32, 
            blocks          = [1,2,3], 
            gestalt_size    = 256,
        ):
        super(UncertaintyPertrainer, self).__init__()

        latent_size = [size[1] // 16, size[0] // 16]

        self.encoder = ConvNeXtEncoder(in_channels, base_channels, blocks=blocks+[0])

        self.pool = MultiArgSequential(
            PositionPooling(latent_size, base_channels*4, gestalt_size),
            Binarize()
        )
        self.gaus2d      = Gaus2D([size[0] // 16, size[1] // 16])
        self.mask_center = MaskCenter(size, combine=True)
            
        self.decoder = MaskDecoder(gestalt_size, base_channels, expand_ratio = 4)

    def forward(self, input, mask):
        position  = self.mask_center(mask)
        gestalt   = self.pool(self.encoder(input), position).unsqueeze(-1).unsqueeze(-1)
    
        position2d = self.gaus2d(position)

        return self.decoder(gestalt * position2d), gestalt, position
