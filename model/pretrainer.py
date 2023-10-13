import torch as th
import torch.nn as nn
import numpy as np
from typing import Tuple
from einops import rearrange, repeat, reduce
from nn.decoder import LociDecoder
from nn.encoder import LociEncoder
from utils.utils import PositionInMask, MultiArgSequential
from utils.loss import MaskedL1SSIMLoss, GestaltLoss, MaskedMSELoss, DecayingMSELoss, DecayingFactor, L1SSIMLoss, MaskedYCbCrL2SSIMLoss, MaskLoss
from nn.mask_decoder import MaskPretrainer, MaskDecoder
from nn.depth_decoder import DepthPretrainer, MaskToDepthDecoder
from nn.rgb_decoder import RGBPretrainer, MaskDepthToRGBDecoder
from nn.downscale import MemoryEfficientPatchDownScale
from nn.upscale import MemoryEfficientUpscaling
from nn.uncertainty_pretrainer import UncertaintyPertrainer
import cv2
from einops.layers.torch import Rearrange

class LociPretrainer(nn.Module):
    def __init__(
        self,
        cfg,
        world_size
    ):
        super(LociPretrainer, self).__init__()

        self.cfg = cfg
        latent_size = [cfg.input_size[0] // 16, cfg.input_size[1] // 16]

        mask_decoder_args = {
            'gestalt_size':    cfg.gestalt_size,
            'mask_channels':   cfg.embedding.mask.channels,
            'expand_ratio':    cfg.embedding.mask.expansion,
        }

        depth_decoder_args = {
            'gestalt_size':    cfg.gestalt_size,
            'num_layers':      cfg.decoder.depth.layers,
            'mask_channels':   cfg.embedding.mask.channels,
            'depth_channels':  cfg.embedding.depth.channels,
            'expand_ratio':    cfg.embedding.depth.expansion,
        }

        rbg_decoder_args = {
            'gestalt_size':    cfg.gestalt_size,
            'num_layers':      cfg.decoder.rgb.layers,
            'mask_channels':   cfg.embedding.mask.channels,
            'depth_channels':  cfg.embedding.depth.channels,
            'rgb_channels':    cfg.embedding.rgb.channels,
            'expand_ratio':    cfg.embedding.rgb.expansion,
        }

        self.encoder = LociEncoder(
            input_size     = cfg.input_size,
            latent_size    = latent_size,
            num_slots      = 1,
            base_channels  = cfg.encoder.channels,
            hyper_channels = cfg.encoder.hyper_channels,
            blocks         = cfg.encoder.blocks,
            gestalt_size   = cfg.gestalt_size,
            batch_size     = cfg.batch_size,
        )

        self.decoder = LociDecoder(
            latent_size   = latent_size,
            num_slots     = 1,
            gestalt_size  = cfg.gestalt_size,
            mask_decoder  = MaskDecoder(**mask_decoder_args),
            depth_decoder = MaskToDepthDecoder(**depth_decoder_args),
            rgb_decoder   = MaskDepthToRGBDecoder(**rbg_decoder_args),
        )

        # TODO futher work
        #self.uncertainty_pretrainer = UncertaintyPertrainer(
        #    size             = cfg.input_size,
        #    in_channels      = 4 if cfg.input_depth else 3,
        #    base_channels    = cfg.uncertainty.channels,
        #    blocks           = cfg.uncertainty.blocks,
        #    gestalt_size     = cfg.gestalt_size,
        #)

        self.mask_pretrainer = MaskPretrainer(
            size             = cfg.input_size,
            encoder_blocks   = cfg.decoder.mask.pretrain_encoder_blocks,
            encoder_channels = cfg.decoder.mask.pretrain_encoder_channels,
            **mask_decoder_args
        )
        self.depth_pretrainer = DepthPretrainer(
            size             = cfg.input_size,
            encoder_blocks   = cfg.decoder.depth.pretrain_encoder_blocks,
            encoder_channels = cfg.decoder.depth.pretrain_encoder_channels,
            **depth_decoder_args
        )
        self.rgb_pretrainer = RGBPretrainer(
            size             = cfg.input_size,
            encoder_blocks   = cfg.decoder.rgb.pretrain_encoder_blocks,
            encoder_channels = cfg.decoder.rgb.pretrain_encoder_channels,
            **rbg_decoder_args
        )

        self.gestalt_mean = nn.Parameter(th.zeros(1, 3*cfg.gestalt_size+1))
        self.gestalt_std  = nn.Parameter(th.ones(1,  3*cfg.gestalt_size+1))
        self.std   = nn.Parameter(th.zeros(1)-1) 
        self.depth = nn.Parameter(th.zeros(1))

        self.compute_position = PositionInMask(cfg.input_size)

        self.mse           = MaskedMSELoss()
        self.maskloss      = nn.MSELoss()
        self.bce_loss      = nn.BCEWithLogitsLoss()
        self.l1ssim        = MaskedL1SSIMLoss() 
        self.rgbloss       = MaskedL1SSIMLoss() if cfg.rgb_loss == 'l1ssim' else MaskedYCbCrL2SSIMLoss()
        self.gestalt_loss  = GestaltLoss()
        self.decying_mse   = DecayingMSELoss(warmup_period = 2 * 10000 * cfg.gradient_accumulation_steps)

        self.init_weight0 = DecayingFactor(warmup_period = 2500 * cfg.gradient_accumulation_steps, min_factor = 0.0)
        self.init_weight1 = DecayingFactor(warmup_period = 5000 * cfg.gradient_accumulation_steps, min_factor = 0.0)
        self.register_buffer('init_counter', th.zeros(1) - 10 * cfg.gradient_accumulation_steps)

        self.mask_patch_embedding = nn.Sequential(
            MemoryEfficientPatchDownScale(1, cfg.embedding.mask.channels, expand_ratio=cfg.embedding.mask.expansion, scale_factor=16),
            nn.Tanh(),
        )

        self.depth_patch_embedding = nn.Sequential(
            MemoryEfficientPatchDownScale(1, cfg.embedding.depth.channels, expand_ratio=cfg.embedding.depth.expansion, scale_factor=16),
            nn.Tanh(),
        )

        self.rgb_patch_embedding = MultiArgSequential(
            MemoryEfficientPatchDownScale(3, cfg.embedding.rgb.channels, expand_ratio=cfg.embedding.rgb.expansion, scale_factor=16),
            nn.Tanh(),
        )

        self.mask_patch_reconstruction  = MemoryEfficientUpscaling(cfg.embedding.mask.channels, 1, expand_ratio=cfg.embedding.mask.expansion, scale_factor=16)
        self.depth_patch_reconstruction = MemoryEfficientUpscaling(cfg.embedding.depth.channels, 1, expand_ratio=cfg.embedding.depth.expansion, scale_factor=16)
        self.rgb_patch_reconstruction   = nn.Sequential(
            MemoryEfficientUpscaling(cfg.embedding.rgb.channels, 3, expand_ratio=cfg.embedding.rgb.expansion, scale_factor=16),
            nn.Sigmoid(),
        )

    def copy_embeddings(self):
        self.mask_pretrainer.load_embeddings(self.mask_patch_embedding, self.mask_patch_reconstruction)
        self.depth_pretrainer.load_embeddings(self.mask_patch_embedding, self.depth_patch_embedding, self.depth_patch_reconstruction)
        self.rgb_pretrainer.load_embeddings(self.mask_patch_embedding, self.depth_patch_embedding, self.rgb_patch_embedding, self.rgb_patch_reconstruction)

    def run_decoder(self, position, gestalt, gt_depth = None, gt_mask = None):

        mask, rgb, _, _, depth_raw = self.decoder(position, gestalt, gt_depth = gt_depth, gt_mask = gt_mask)
        mask  = th.softmax(th.cat((mask, th.ones_like(mask)), dim=1), dim=1)[:,:1] 

        return mask, rgb, depth_raw

    def pretraining_stage0_embedding(self, input_rgb, input_depth, input_instance_mask):
        
        confidence          = input_instance_mask.clone().detach()
        input_instance_mask = (input_instance_mask > 0.5).float().detach()
        depth_weight        = (input_depth >= 0).float().detach() * confidence.detach()

        B = input_rgb.shape[0]
        _, gt_xy, gt_std = self.compute_position(input_instance_mask)       
        z   = th.zeros_like(gt_std)

        position = th.cat((gt_xy, z, gt_std), dim=1) 
        gestalt  = th.zeros((B, 3*self.cfg.gestalt_size+1), device=input_rgb.device)

        input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
        input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        input_depth_std  = th.sqrt(
            th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
            (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        )

        input_depth = ((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask
        input_rgb   = input_rgb * input_instance_mask

        mask = self.mask_patch_reconstruction(self.mask_patch_embedding(input_instance_mask))
        mask = th.softmax(th.cat((mask, th.ones_like(mask)), dim=1), dim=1)[:,:1]
        
        with th.no_grad():
            intersection = th.sum((mask > 0.5).float() * input_instance_mask, dim=(1,2,3))
            union        = th.sum(th.maximum((mask > 0.5).float(), input_instance_mask), dim=(1,2,3))
            mean_iou     = th.mean(intersection / (union + 1e-6))

        depth = self.depth_patch_reconstruction(self.depth_patch_embedding(input_depth))
        rgb   = self.rgb_patch_reconstruction(self.rgb_patch_embedding(input_rgb))

        depth_loss, depth_l1, depth_ssim = self.l1ssim(depth, input_depth, depth_weight)
        rgb_loss, rgb_l1, rgb_ssim       = self.rgbloss(rgb, input_rgb, confidence)

        return {
            'mask_loss'     : self.maskloss(mask, input_instance_mask),
            'mean_iou'      : mean_iou,
            'rgb_loss'      : rgb_loss,
            'depth_loss'    : depth_loss,
            'position_loss' : 0,
            'z_loss'        : 0,
            'gestalt_loss'  : 0,
            'rgb_l1'        : rgb_l1.detach(),
            'depth_l1'      : depth_l1.detach(),
            'rgb_ssim'      : rgb_ssim.detach(),
            'depth_ssim'    : depth_ssim.detach(),
            'mask'          : mask,
            'rgb'           : rgb,
            'depth'         : depth,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : th.tensor([0, 0, 0], device=input_instance_mask.device),
        }

    def pretraining_stage0_uncertainty(self, input_rgb, input_depth, input_instance_mask):
        
        input_instance_mask = (input_instance_mask > 0.5).float().detach()
        input = th.cat((input_rgb, input_depth), dim=1) if self.cfg.input_depth else input_rgb

        mask, gestalt, position = self.uncertainty_pretrainer(input, input_instance_mask)
        
        with th.no_grad():
            _mask = (th.sigmoid(mask) > 0.5).float()
            intersection = th.sum(_mask * input_instance_mask, dim=(1,2,3))
            union        = th.sum(th.maximum(_mask, input_instance_mask), dim=(1,2,3))
            mean_iou     = th.mean(intersection / (union + 1e-6))

        return {
            'mask_loss'     : self.bce_loss(mask, input_instance_mask),
            'mean_iou'      : mean_iou,
            'rgb_loss'      : 0,
            'depth_loss'    : 0,
            'position_loss' : 0,
            'z_loss'        : 0,
            'gestalt_loss'  : 0,
            'rgb_l1'        : 0,
            'depth_l1'      : 0,
            'rgb_ssim'      : 0,
            'depth_ssim'    : 0,
            'mask'          : mask,
            'rgb'           : None,
            'depth'         : None,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : th.tensor([0, 0, 0], device=input_instance_mask.device),
        }


    def pretraining_stage0_mask(self, input_instance_mask):
        
        confidence = input_instance_mask.clone().detach()
        input_instance_mask = (input_instance_mask > 0.5).float().detach()

        mask, gestalt, position = self.mask_pretrainer(input_instance_mask)
        mask = th.softmax(th.cat((mask, th.ones_like(mask)), dim=1), dim=1)[:,:1]
        
        with th.no_grad():
            intersection = th.sum((mask > 0.5).float() * input_instance_mask, dim=(1,2,3))
            union        = th.sum(th.maximum((mask > 0.5).float(), input_instance_mask), dim=(1,2,3))
            mean_iou     = th.mean(intersection / (union + 1e-6))

        return {
            'mask_loss'     : self.maskloss(mask, input_instance_mask),
            'mean_iou'      : mean_iou,
            'rgb_loss'      : 0,
            'depth_loss'    : 0,
            'position_loss' : 0,
            'z_loss'        : 0,
            'gestalt_loss'  : 0,
            'rgb_l1'        : 0,
            'depth_l1'      : 0,
            'rgb_ssim'      : 0,
            'depth_ssim'    : 0,
            'mask'          : mask,
            'rgb'           : None,
            'depth'         : None,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : th.tensor([0, 0, 0], device=input_instance_mask.device),
        }

    def pretraining_stage0_depth(self, input_depth, input_instance_mask):
        
        confidence = input_instance_mask.clone().detach()
        input_instance_mask = (input_instance_mask > 0.5).float().detach()
        depth_weight = (input_depth >= 0).float().detach() * confidence.detach()

        input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
        input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        input_depth_std  = th.sqrt(
            th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
            (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        )

        input_depth = ((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask

        depth, gestalt, position = self.depth_pretrainer(input_instance_mask, input_depth)

        depth_loss, depth_l1, depth_ssim = self.l1ssim(depth, input_depth, depth_weight)

        return {
            'mask_loss'     : 0,
            'mean_iou'      : 0,
            'rgb_loss'      : 0,
            'depth_loss'    : depth_loss,
            'position_loss' : 0,
            'z_loss'        : 0,
            'gestalt_loss'  : 0,
            'rgb_l1'        : 0,
            'depth_l1'      : depth_l1.detach(),
            'rgb_ssim'      : 0,
            'depth_ssim'    : depth_ssim.detach(),
            'mask'          : None,
            'rgb'           : None,
            'depth'         : depth,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : th.tensor([0, 0, 0], device=input_instance_mask.device),
        }

    def pretraining_stage0_rgb(self, input_rgb, input_depth, input_instance_mask):
        
        confidence = input_instance_mask.clone().detach()
        input_instance_mask = (input_instance_mask > 0.5).float().detach()

        input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
        input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        input_depth_std  = th.sqrt(
            th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
            (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        )

        input_depth = ((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask
        input_rgb   = input_rgb * input_instance_mask

        rgb, gestalt, position = self.rgb_pretrainer(input_instance_mask, input_depth, input_rgb)

        rgb_loss, rgb_l1, rgb_ssim = self.rgbloss(rgb, input_rgb, confidence)

        return {
            'mask_loss'     : 0,
            'mean_iou'      : 0,
            'rgb_loss'      : rgb_loss,
            'depth_loss'    : 0,
            'position_loss' : 0, 
            'z_loss'        : 0,
            'gestalt_loss'  : 0, #self.gestalt_loss(gestalt),
            'rgb_l1'        : rgb_l1.detach(),
            'depth_l1'      : 0,
            'rgb_ssim'      : rgb_ssim.detach(),
            'depth_ssim'    : 0,
            'mask'          : None,
            'rgb'           : rgb,
            'depth'         : None,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : th.tensor([0, 0, 0], device=input_instance_mask.device),
        }

    def pretraining_stage1(self, input_rgb, input_depth, input_instance_mask):
        B, _, H, W = input_rgb.shape
        device     = input_rgb.device

        xy_init, gt_xy, gt_std = self.compute_position(input_instance_mask)       
        std = repeat(self.std, '1 -> b 1', b = B)
        z   = repeat(self.depth, '1 -> b 1', b = B)

        position = th.cat((xy_init, z, std), dim=1) 
        gestalt  = th.zeros((B, 3*self.cfg.gestalt_size+1), device=device)

        confidence          = input_instance_mask.clone().detach()
        input_instance_mask = (input_instance_mask > 0.5).float().detach()
        depth_weight        = (input_depth >= 0).float().detach() * confidence.detach()

        input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
        input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        input_depth_std  = th.sqrt(
            th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
            (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        )

        gt_depth   = ((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask
        gt_z       = reduce(input_depth * input_instance_mask, 'b 1 h w -> b 1', 'sum') / (reduce(input_instance_mask, 'b 1 h w -> b 1', 'sum') + 1e-8)
        gt_z_scale = rearrange(input_depth_std, 'b 1 1 1 -> b 1')
        z_weight   = reduce(depth_weight, 'b 1 h w -> b 1', 'max')

        gt_position = th.cat((gt_xy, gt_z, gt_std), dim=1)
        
        position, gestalt, _ = self.encoder(
            input_rgb         = input_rgb, 
            input_depth       = input_depth if self.cfg.input_depth else th.zeros_like(input_depth),
            error_last        = th.zeros_like(input_depth),
            mask              = th.zeros_like(th.cat((input_instance_mask, th.zeros_like(input_instance_mask)), dim=1)), 
            mask_raw          = th.zeros_like(input_instance_mask),
            slot_rgb          = th.zeros_like(input_rgb),
            slot_depth        = th.zeros_like(input_depth),
            slot_flow         = th.zeros((B, 2, H, W), device=device), 
            position          = position, 
            gestalt           = gestalt,
            slot_reset        = th.zeros_like(position[:,:1]),
            use_hyper_weights = False,
        )
        mask, rgb, depth = self.run_decoder(
            gt_position, gestalt, gt_depth = gt_depth, gt_mask = input_instance_mask
        )

        with th.no_grad():
            intersection = th.sum((mask > 0.5).float() * input_instance_mask, dim=(1,2,3))
            union        = th.sum(th.maximum((mask > 0.5).float(), input_instance_mask), dim=(1,2,3))
            mean_iou     = th.mean(intersection / (union + 1e-6))

        z_scale = gestalt[:, -1:]  
        z       = position[:,2:3]

        rgb_loss, rgb_l1, rgb_ssim       = self.rgbloss(rgb, input_rgb, confidence)
        depth_loss, depth_l1, depth_ssim = self.l1ssim(depth, gt_depth, depth_weight)

        mask_loss     = self.maskloss(mask, input_instance_mask)
        position_loss = th.mean((position[:,:2]  - gt_xy)**2)           
        position_loss = position_loss + th.mean((position[:,-1:] - gt_std)**2)          
        z_loss        = th.mean(((z - gt_z)**2) * z_weight)             
        z_loss        = z_loss + th.mean(((z_scale - gt_z_scale)**2) * z_weight) 

        return {
            'mask_loss'     : mask_loss*100,
            'mean_iou'      : mean_iou,
            'rgb_loss'      : rgb_loss,
            'depth_loss'    : depth_loss,
            'position_loss' : position_loss,
            'z_loss'        : z_loss,
            'gestalt_loss'  : 0,
            'rgb_l1'        : rgb_l1.detach(),
            'depth_l1'      : depth_l1.detach(),
            'rgb_ssim'      : rgb_ssim.detach(),
            'depth_ssim'    : depth_ssim.detach(),
            'mask'          : mask,
            'rgb'           : rgb,
            'depth'         : depth,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : th.tensor([0, 0, 0], device=input_instance_mask.device),
        }

    def pretraining_stage2(self, input_rgb, input_depth, input_instance_mask, iterations = 3):
        assert iterations <= 3
        B, _, H, W = input_rgb.shape
        device     = input_rgb.device

        self.init_counter = (self.init_counter + 1).detach()
        init_weight0      = self.init_weight0.get()
        init_weight1      = self.init_weight1.get()
        time_weight       = th.tensor([max(init_weight0, 0.01), min(1 - init_weight0, max(0.1, init_weight1)), 1 - init_weight1], device = device)
        time_weight       = time_weight / time_weight.sum()

        xy_init, gt_xy, gt_std = self.compute_position(input_instance_mask)       
        std = repeat(self.std, '1 -> b 1', b = B)
        z   = repeat(self.depth, '1 -> b 1', b = B)

        position = th.cat((xy_init, z, std), dim=1) 
        gestalt  = th.zeros((B, 3*self.cfg.gestalt_size+1), device=device)

        confidence          = input_instance_mask.clone().detach()
        input_instance_mask = (input_instance_mask > 0.5).float().detach()
        depth_weight        = (input_depth >= 0).float().detach() * confidence.detach()

        input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
        input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        input_depth_std  = th.sqrt(
            th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
            (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
        )

        gt_depth   = ((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask
        gt_z       = reduce(input_depth * input_instance_mask, 'b 1 h w -> b 1', 'sum') / (reduce(input_instance_mask, 'b 1 h w -> b 1', 'sum') + 1e-8)
        gt_z_scale = rearrange(input_depth_std, 'b 1 1 1 -> b 1')
        z_weight   = reduce(depth_weight, 'b 1 h w -> b 1', 'max')

        gt_position = th.cat((gt_xy, gt_z, gt_std), dim=1)

        mean_iou = mask_loss = rgb_loss = depth_loss = position_loss = z_loss = 0
        rgb_l1 = rgb_ssim = depth_l1 = depth_ssim = None

        mask  = th.zeros_like(input_instance_mask)
        depth = th.zeros_like(input_depth)
        rgb   = th.zeros_like(input_rgb)

        for t in range(iterations):

            # make shure we do a few iterattions full loop so that we directly crash if we don't have enough memory
            if time_weight[t] == 0 and self.init_counter.item() > 0:
                continue

            position, gestalt, _ = self.encoder(
                input_rgb         = input_rgb, 
                input_depth       = input_depth if self.cfg.input_depth else th.zeros_like(input_depth),
                error_last        = th.zeros_like(input_depth),
                mask              = th.cat((mask, th.zeros_like(mask)), dim=1),
                mask_raw          = th.zeros_like(input_instance_mask),
                slot_rgb          = rgb * mask,
                slot_depth        = depth * mask,
                slot_flow         = th.zeros_like(input_rgb[:,:2]),
                position          = position, 
                gestalt           = gestalt,
                use_hyper_weights = (t > 0),
            )

            mask, rgb, depth = self.run_decoder(position, gestalt)

            with th.no_grad():
                intersection = th.sum((mask > 0.5).float() * input_instance_mask, dim=(1,2,3))
                union        = th.sum(th.maximum((mask > 0.5).float(), input_instance_mask), dim=(1,2,3))
                mean_iou     = mean_iou + th.mean(intersection / (union + 1e-6)) * time_weight[t]

            z_scale = gestalt[:, -1:]  
            z       = position[:,2:3]

            _rgb_loss, rgb_l1, rgb_ssim       = self.rgbloss(rgb, input_rgb, confidence)
            _depth_loss, depth_l1, depth_ssim = self.l1ssim(depth, gt_depth, depth_weight)

            mask_loss     = mask_loss     + self.maskloss(mask, input_instance_mask)        * time_weight[t]
            rgb_loss      = rgb_loss      + _rgb_loss                                       * time_weight[t]
            depth_loss    = depth_loss    + _depth_loss                                     * time_weight[t]
            position_loss = position_loss + self.decying_mse(position[:,:2], gt_xy)         * time_weight[t]
            position_loss = position_loss + self.decying_mse(position[:,-1:], gt_std)       * time_weight[t]
            z_loss        = z_loss        + th.mean(((z - gt_z)**2) * z_weight)             * time_weight[t]
            z_loss        = z_loss        + th.mean(((z_scale - gt_z_scale)**2) * z_weight) * time_weight[t]

        return {
            'mask_loss'     : mask_loss*100,
            'mean_iou'      : mean_iou,
            'rgb_loss'      : rgb_loss,
            'depth_loss'    : depth_loss,
            'position_loss' : position_loss,
            'z_loss'        : z_loss,
            'gestalt_loss'  : 0,
            'rgb_l1'        : rgb_l1.detach(),
            'depth_l1'      : depth_l1.detach(),
            'rgb_ssim'      : rgb_ssim.detach(),
            'depth_ssim'    : depth_ssim.detach(),
            'mask'          : mask,
            'rgb'           : rgb,
            'depth'         : depth,
            'position'      : position,
            'gestalt'       : gestalt,
            'time_weight'   : time_weight,
        }

    def forward(self, input_rgb, input_depth, input_instance_mask, iterations = 3, mode='all'):
        if mode == 'embedding':
            return self.pretraining_stage0_embedding(input_rgb, input_depth, input_instance_mask)

        if mode == 'mask':
            return self.pretraining_stage0_mask(input_instance_mask)
        if mode == 'depth':
            return self.pretraining_stage0_depth(input_depth, input_instance_mask)
        if mode == 'rgb':
            return self.pretraining_stage0_rgb(input_rgb, input_depth, input_instance_mask)
        if mode == 'uncertainty':
            return self.pretraining_stage0_uncertainty(input_rgb, input_depth, input_instance_mask)

        if iterations == 0:
            return self.pretraining_stage1(input_rgb, input_depth, input_instance_mask)

        return self.pretraining_stage2(input_rgb, input_depth, input_instance_mask, iterations)
