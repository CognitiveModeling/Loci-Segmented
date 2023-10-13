import torch as th
import torch.nn as nn
import numpy as np
from typing import Tuple
from einops import rearrange, repeat, reduce
from nn.decoder import LociDecoder
from nn.encoder import LociEncoder
from nn.predictor import LatentEpropPredictor, UpdateModule
from utils.utils import SharedObjectsToBatch, BatchToSharedObjects, RadomSimilarityBasedMaskDrop
from utils.loss import MaskModulatedObjectLoss, ObjectModulator, TranslationInvariantObjectLoss, PositionLoss
from nn.object_discovery import ObjectDiscovery
from nn.background import UncertantyBackground
from nn.mask_decoder import MaskDecoder
from nn.depth_decoder import MaskToDepthDecoder
from nn.rgb_decoder import MaskDepthToRGBDecoder
from nn.proposal import LociProposal

class Loci(nn.Module):
    def __init__(self, cfg):
        super(Loci, self).__init__()

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
            num_slots      = cfg.num_slots,
            base_channels  = cfg.encoder.channels,
            hyper_channels = cfg.encoder.hyper_channels,
            blocks         = cfg.encoder.blocks,
            gestalt_size   = cfg.gestalt_size,
            batch_size     = cfg.batch_size,
        )

        self.decoder = LociDecoder(
            latent_size   = latent_size,
            num_slots     = cfg.num_slots,
            gestalt_size  = cfg.gestalt_size,
            mask_decoder  = MaskDecoder(**mask_decoder_args),
            depth_decoder = MaskToDepthDecoder(**depth_decoder_args),
            rgb_decoder   = MaskDepthToRGBDecoder(**rbg_decoder_args),
        )


        self.predictor = LatentEpropPredictor(
            num_slots    = cfg.num_slots,
            gestalt_size = cfg.gestalt_size,
            heads        = cfg.predictor.heads,
            layers       = cfg.predictor.layers,
            reg_lambda   = cfg.predictor.reg_lambda,
            batch_size   = cfg.batch_size,
            num_hidden   = cfg.predictor.channels,
        )

        self.update_gate = UpdateModule(
            gestalt_size     = cfg.gestalt_size,
            num_hidden       = cfg.predictor.gate.num_hidden,
            num_layers       = cfg.predictor.gate.num_layers,
            num_slots        = cfg.num_slots,
            gate_noise_level = cfg.predictor.gate.noise_level,
            reg_lambda       = cfg.predictor.gate.reg_lambda,
        )


        self.object_discovery = ObjectDiscovery(
            gestalt_size               = cfg.gestalt_size * 3 + 1,
            num_slots                  = cfg.num_slots,
            object_permanence_strength = cfg.object_permanence_strength,
            entity_pretraining_steps   = cfg.entity_pretraining_steps
        )

        self.background = UncertantyBackground(
            uncertainty_threshold   = cfg.background.uncertainty_threshold,
            motion_context_size     = cfg.background.motion_context_size,
            depth_context_size      = cfg.background.depth_context_size,
            latent_channels         = cfg.background.latent_channels,
            num_layers              = cfg.background.num_layers,
            depth_input             = cfg.input_depth,
        )

        self.translation_invariant_object_loss = TranslationInvariantObjectLoss(cfg.num_slots, cfg.teacher_forcing)
        self.mask_modulated_object_loss        = MaskModulatedObjectLoss(cfg.num_slots, cfg.teacher_forcing)
        self.position_loss                     = PositionLoss(cfg.num_slots, cfg.teacher_forcing)
        self.modulator                         = ObjectModulator(cfg.num_slots)

        self.to_batch  = SharedObjectsToBatch(cfg.num_slots)
        self.to_shared = BatchToSharedObjects(cfg.num_slots)

        self.mask_drop = RadomSimilarityBasedMaskDrop()

        self.mse_loss = nn.MSELoss()

        if cfg.inference_mode == "segmentation":
            self.proposal = LociProposal(
                size           = cfg.input_size,
                num_slots      = cfg.position_proposal.num_slots,
                encoder_blocks = cfg.position_proposal.encoder_blocks,
                decoder_blocks = cfg.position_proposal.decoder_blocks,
                base_channels  = cfg.position_proposal.base_channels,
                depth_input    = cfg.input_depth,
            )

    def get_init_status(self):
        return self.background.get_init()

    def get_openings(self):
        return self.predictor.get_openings()

    def detach(self):
        for module in self.modules():
            if module != self and callable(getattr(module, "detach", None)):
                module.detach()

    def reset_state(self):
        for module in self.modules():
            if module != self and callable(getattr(module, "reset_state", None)):
                module.reset_state()

    def run_decoder(
        self, 
        position: th.Tensor, 
        gestalt: th.Tensor,
        priority: th.Tensor,
        bg_rgb: th.Tensor,
        bg_depth: th.Tensor,
        compute_raw: bool,
        teacher_forcing = False
    ):
        mask, rgb, depth, mask_raw, depth_raw = self.decoder(
            position, gestalt, priority, compute_raw = compute_raw
        )
        
        if teacher_forcing and self.cfg.inference_mode == "regularization":
            mask = self.mask_drop(position, gestalt, mask)

        mask  = th.softmax(th.cat((mask, th.ones_like(mask[:,:1])), dim=1), dim=1) 
        rgb   = th.cat((rgb, bg_rgb), dim=1)
        depth = th.cat((depth, bg_depth), dim=1)

        occlusion = th.zeros((mask.shape[0], mask.shape[1]-1), device = mask.device)
        if mask_raw is not None:
            mask_raw = self.to_batch(mask_raw)
            mask_raw = self.to_shared(th.softmax(th.cat((mask_raw, th.ones_like(mask_raw)), dim=1), dim=1)[:,:1])

            occlusion = th.sum((mask[:,:-1] > 0.8).float(), dim=(2,3)) / (th.sum(mask_raw > 0.8, dim=(2,3)) + 1e-8)
            occlusion = th.clamp(1 - occlusion, 0, 1)

        _mask  = mask.unsqueeze(dim=2)
        _rgb   = rearrange(rgb, 'b (o c) h w -> b o c h w', o = self.cfg.num_slots+1)
        _depth = rearrange(depth,  'b o h w -> b o 1 h w', o = self.cfg.num_slots+1)

        output_rgb   = th.sum(_mask * _rgb, dim=1)
        output_depth = th.sum(_mask * _depth, dim=1)
        return {
            "output_rgb"   : output_rgb, 
            "output_depth" : output_depth, 
            "mask"         : mask, 
            "rgb"          : rgb, 
            "depth"        : depth, 
            "mask_raw"     : mask_raw, 
            "depth_raw"    : depth_raw,
            "occlusion"    : occlusion
        }

    def forward(
        self, 
        input_rgb       : th.Tensor,
        input_depth     : th.Tensor,
        bg_rgb_last     : th.Tensor = None, 
        bg_depth_last   : th.Tensor = None,
        rgb_last        : th.Tensor = None, 
        depth_raw_last  : th.Tensor = None,
        mask_last       : th.Tensor = None,
        mask_raw_last   : th.Tensor = None,
        occlusion_last  : th.Tensor = None,
        position_last   : th.Tensor = None,
        gestalt_last    : th.Tensor = None,
        priority_last   : th.Tensor = None,
        teacher_forcing : bool      = False,
        reset    = True, 
        detach   = True, 
        evaluate = False, 
        test     = False,
    ):
        if detach:
            self.detach()

        if reset:
            self.reset_state()

        init_mask = mask_last.clone() if mask_last is not None else None

        position_loss    = th.tensor(0, device=input_rgb.device)
        object_loss      = th.tensor(0, device=input_rgb.device)
        time_loss        = th.tensor(0, device=input_rgb.device)
        uncertainty_loss = th.tensor(0, device=input_rgb.device)
        output_cur       = {}
        output_next      = {}

        # compute background and uncertainty
        bg_input        = th.cat((input_rgb, input_depth), dim=1) if self.cfg.input_depth else input_rgb
        uncertainty_cur = self.background.uncertainty_estimation(bg_input)[0]
        bg_rgb, bg_depth, _, depth_context = self.background(bg_input, bg_input, uncertainty_cur, uncertainty_cur, delta_t = th.zeros_like(uncertainty_cur[:,0:1,0:1,0:1]), color_input= th.ones_like(input_rgb[:,0:1,0:1,0:1]))

        error_last = uncertainty_cur if mask_last is None else th.relu(uncertainty_cur - th.sum(mask_last[:,:-1], dim=1, keepdim=True))

        position_last, gestalt_last, priority_last, slot_reset = self.object_discovery(
            error_last, mask_last, position_last, gestalt_last, priority_last
        )

        # compute the unprioritized outputs for the first timepoint in a thruncated backpropagation through time sequence
        if rgb_last is None or depth_raw_last is None or input_depth is None or mask_raw_last is None or occlusion_last is None:
            output_last = self.run_decoder(
                position    = position_last, 
                gestalt     = gestalt_last, 
                priority    = None,
                bg_rgb      = bg_rgb_last   if bg_rgb_last   is not None else bg_rgb,
                bg_depth    = bg_depth_last if bg_depth_last is not None else bg_depth,
                compute_raw = True,
                teacher_forcing = teacher_forcing
            )

            if rgb_last is None:
                rgb_last = output_last['rgb']

            if depth_raw_last is None:
                depth_raw_last = output_last['depth_raw']

            if input_depth is None:
                input_depth = output_last['output_depth']

            if mask_raw_last is None:
                mask_raw_last = output_last['mask_raw']

            if occlusion_last is None:
                occlusion_last = output_last['occlusion']

        if mask_last is None:
            mask_last = th.zeros((input_rgb.shape[0], self.cfg.num_slots+1, *input_rgb.shape[2:]), device=input_rgb.device)

        error_last = th.relu(uncertainty_cur - th.sum(mask_last[:,:-1], dim=1, keepdim=True)) 

        # position and gestalt for the current time point
        position_cur, gestalt_cur, priority_cur = self.encoder(
            input_rgb         = input_rgb, 
            input_depth       = input_depth,
            error_last        = error_last, 
            mask              = mask_last, 
            mask_raw          = mask_raw_last,
            slot_rgb          = rgb_last,
            slot_depth        = depth_raw_last,
            slot_flow         = th.zeros((input_rgb.shape[0], 2 * self.cfg.num_slots, *input_rgb.shape[2:]), device=input_rgb.device), # TODO use flow
            position          = position_last, 
            gestalt           = gestalt_last,
            slot_reset        = slot_reset,
            use_hyper_weights = not reset
        )

        output_cur = self.run_decoder(
            position    = position_cur,
            gestalt     = gestalt_cur, 
            priority    = priority_cur,
            bg_rgb      = bg_rgb_last   if bg_rgb_last   is not None else bg_rgb, 
            bg_depth    = bg_depth_last if bg_depth_last is not None else bg_depth,
            compute_raw = True,
            teacher_forcing = teacher_forcing
        )

        # position and gestalt for the next time point
        if not teacher_forcing:

            (
                position_cur,  gestalt_cur,  priority_cur, 
                position_gate, gestalt_gate
            ) = self.update_gate(
                position_cur,  gestalt_cur,  priority_cur,  output_cur['occlusion'], 
                position_last, gestalt_last, priority_last, occlusion_last
            )

            position_next, gestalt_next, priority_next = self.predictor(
                position_cur, gestalt_cur, priority_cur
            ) 

            # combinded background and lots for next timepoint
            output_next = self.run_decoder(
                position    = position_next, 
                gestalt     = gestalt_next, 
                priority    = priority_next, 
                bg_rgb      = bg_rgb, 
                bg_depth    = bg_depth, 
                compute_raw = not evaluate and not test
            )
        elif self.cfg.inference_mode == 'segmentation':
            output_cur['mask'] = init_mask


        if not evaluate and not test:

            #regularize to small possition chananges over time
            if not teacher_forcing:
                position_loss = position_loss + self.position_loss(position_next, position_last.detach(), mask_last[:,:-1].detach())

                # regularize to encode last visible object
                # TODO add depth 
                if self.cfg.object_regularizer > 0:
                    object_next_unprioritized_modulated = self.decoder(*self.modulator(
                        position_next, gestalt_next, output_next['mask'][:,:-1])
                    )[-2]
                    object_loss = object_loss + self.mask_modulated_object_loss(
                        output_next['object'], 
                        object_next_unprioritized_modulated.detach(), 
                        output_next['mask'][:,:-1].detach()
                    )

                # regularize to produce consistent object codes over time
                # TODO add depth 
                time_loss = time_loss + 0.1 * self.translation_invariant_object_loss(
                    output_next['mask'][:,:-1].detach(),
                    rgb_last[:,:-3].detach(),
                    position_last.detach(),
                    output_next['rgb'][:,:-3], 
                    position_next.detach(),
                )

            if mask_last is not None:
                num_used_slots = reduce((mask_last[:,:-1] > 0.75).float(), 'b c h w -> b c 1 1', 'max').detach()
                num_used_slots = th.mean(num_used_slots, dim=1, keepdim=True)
                uncertainty_loss = uncertainty_loss + self.mse_loss(uncertainty_cur, th.sum(mask_last[:,:-1], dim=1, keepdim=True)) * num_used_slots

        output = {
            "reconstruction" : {
                **output_cur,
                "position"      : position_cur,
                "gestalt"       : gestalt_cur,
                "priority"      : priority_cur,
                "uncertainty"   : uncertainty_cur,
            },
            "prediction"   : {
                "bg_rgb"   : bg_rgb, 
                "bg_depth" : bg_depth, 
            },
            "position_loss"    : position_loss,
            "object_loss"      : object_loss,
            "time_loss"        : time_loss,
            "uncertainty_loss" : uncertainty_loss,
        }
        if not teacher_forcing:
            output['prediction'].update({
                **output_next,
                "position"      : position_next, 
                "gestalt"       : gestalt_next, 
                "priority"      : priority_next, 
                "position_gate" : position_gate,
                "gestalt_gate"  : gestalt_gate,
            })

        return output
