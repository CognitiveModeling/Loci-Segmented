import pytorch_lightning as pl
import torch as th
import numpy as np
from torch.utils.data import DataLoader
from utils.optimizers import Ranger
from utils.configuration import Configuration
from model.pretrainer import LociPretrainer
from utils.io import UEMA, Timer
import torch.distributed as dist
from einops import rearrange, repeat, reduce

def check_state_dicts(name, state_dict_a, state_dict_b):
    # check for missing keys
    for k in state_dict_a.keys():
        if k not in state_dict_b:
            print(f"WARNING: missing key {k} in {name} checkpoint")

    # check for extra keys
    for k in state_dict_b.keys():
        if k not in state_dict_a:
            print(f"WARNING: extra key {k} in {name} checkpoint")

    # check for shape mismatches
    for k in state_dict_a.keys():
        #if k in self.net.decoder.to_depth.state_dict() and state_dict['depth_decoder'][k].shape != self.net.decoder.to_depth.state_dict()[k].shape:
        if k in state_dict_b and state_dict_a[k].shape != state_dict_b[k].shape:
            print(f"WARNING: shape mismatch for key {k} in {name} checkpoint ({state_dict_a[k].shape} vs {state_dict_b[k].shape})")


class LociPretrainerModule(pl.LightningModule):
    def __init__(self, cfg: Configuration, state_dict={}):
        super().__init__()
        self.cfg = cfg

        print(f"RANDOM SEED: {cfg.seed}")
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        self.net = LociPretrainer(self.cfg.model, cfg.world_size)

        print(f"Parameters:                            {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print(f"encoder Parameters:                    {sum(p.numel() for p in self.net.encoder.parameters() if p.requires_grad)}")
        print(f"decoder Parameters:                    {sum(p.numel() for p in self.net.decoder.parameters() if p.requires_grad)}")
        print(f"mask_pretrainer Parameters:            {sum(p.numel() for p in self.net.mask_pretrainer.parameters() if p.requires_grad)}")
        print(f"depth_pretrainer Parameters:           {sum(p.numel() for p in self.net.depth_pretrainer.parameters() if p.requires_grad)}")
        print(f"rgb_pretrainer Parameters:             {sum(p.numel() for p in self.net.rgb_pretrainer.parameters() if p.requires_grad)}")
        print(f"mask_decoder Parameters:               {sum(p.numel() for p in self.net.decoder.to_mask.parameters() if p.requires_grad)}")
        print(f"depth_decoder Parameters:              {sum(p.numel() for p in self.net.decoder.to_depth.parameters() if p.requires_grad)}")
        print(f"rgb_decoder Parameters:                {sum(p.numel() for p in self.net.decoder.to_rgb.parameters() if p.requires_grad)}")
        print(f"mask_patch_embedding Parameters:       {sum(p.numel() for p in self.net.mask_patch_embedding.parameters() if p.requires_grad)}")
        print(f"depth_patch_embedding Parameters:      {sum(p.numel() for p in self.net.depth_patch_embedding.parameters() if p.requires_grad)}")
        print(f"rgb_patch_embedding Parameters:        {sum(p.numel() for p in self.net.rgb_patch_embedding.parameters() if p.requires_grad)}")
        print(f"mask_patch_reconstruction Parameters:  {sum(p.numel() for p in self.net.mask_patch_reconstruction.parameters() if p.requires_grad)}")
        print(f"depth_patch_reconstruction Parameters: {sum(p.numel() for p in self.net.depth_patch_reconstruction.parameters() if p.requires_grad)}")
        print(f"rgb_patch_reconstruction Parameters:   {sum(p.numel() for p in self.net.rgb_patch_reconstruction.parameters() if p.requires_grad)}")


        self.lr = self.cfg.learning_rate
        self.own_loggers = {}
        self.timer = Timer()
        self.val_metrics = {}
        self.world_size_checked = False

        for param in self.net.encoder.priority_encoder.parameters():
            param.requires_grad_(False)

        if 'mask_decoder' in state_dict:
            check_state_dicts('mask_decoder', state_dict['mask_decoder'], self.net.decoder.to_mask.state_dict())
            self.net.decoder.to_mask.load_state_dict(state_dict['mask_decoder'], strict=False)
            print("Loaded mask decoder")

        if 'mask_pretrainer' in state_dict:
            check_state_dicts('mask_pretrainer', state_dict['mask_pretrainer'], self.net.mask_pretrainer.state_dict())
            self.net.mask_pretrainer.load_state_dict(state_dict['mask_pretrainer'], strict=False)
            print("Loaded mask pretrainer")

        if 'depth_decoder' in state_dict:
            check_state_dicts('depth_decoder', state_dict['depth_decoder'], self.net.decoder.to_depth.state_dict())
            self.net.decoder.to_depth.load_state_dict(state_dict['depth_decoder'], strict=False)
            print("Loaded depth decoder")

        if 'depth_pretrainer' in state_dict:
            check_state_dicts('depth_pretrainer', state_dict['depth_pretrainer'], self.net.depth_pretrainer.state_dict())
            self.net.depth_pretrainer.load_state_dict(state_dict['depth_pretrainer'], strict=False)
            print("Loaded depth pretrainer")

        if 'rgb_decoder' in state_dict:
            check_state_dicts('rgb_decoder', state_dict['rgb_decoder'], self.net.decoder.to_rgb.state_dict())
            self.net.decoder.to_rgb.load_state_dict(state_dict['rgb_decoder'], strict=False)
            print("Loaded rgb decoder")

        if 'rgb_pretrainer' in state_dict:
            check_state_dicts('rgb_pretrainer', state_dict['rgb_pretrainer'], self.net.rgb_pretrainer.state_dict())
            self.net.rgb_pretrainer.load_state_dict(state_dict['rgb_pretrainer'], strict=False)
            print("Loaded rgb pretrainer")

        #if cfg.pretrainer_iterations == 0:
        #    for param in self.net.decoder.parameters():
        #        param.requires_grad_(False)

        self.num_updates = -1

    def forward(self, input_rgb, input_depth, input_instance_mask):

        if not self.world_size_checked:
            if dist.get_world_size() != self.cfg.world_size:
                print(f"WOLRD SIZE CHECK FAILED!!! Expected world size of {self.cfg.world_size}, but got {dist.get_world_size()}")
            else:
                self.world_size_checked = True

        return self.net(input_rgb, input_depth, input_instance_mask, iterations=self.cfg.pretrainer_iterations, mode=self.cfg.pretraining_mode)

    def log(self, name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True):
        super().log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

        if name.startswith("val_"):
            if name not in self.val_metrics:
                self.val_metrics[name] = 0
                print("Adding metric: ", name)

            self.val_metrics[name] += value.item() if isinstance(value, th.Tensor) else value
        else:
            if name not in self.own_loggers:
                self.own_loggers[name] = UEMA(1000)

            self.own_loggers[name].update(value.item() if isinstance(value, th.Tensor) else value)

    def get_binarization_stats(self, gestalt, capacity):

        intervall = th.linspace(0, 1, gestalt.shape[-1], device=gestalt.device)
        mask      = (intervall < capacity).float().expand_as(gestalt)

        binarization      = th.min(th.abs(gestalt), th.abs(1 - gestalt)).detach()
        binarization_mean = th.sum(binarization * mask) / (th.sum(mask) + 1e-16)
        binariation_std   = th.sqrt(th.sum(((binarization - binarization_mean) ** 2) * mask) / (th.sum(mask) + 1e-16))
        gestalt_mean      = th.sum(th.clip(gestalt, 0, 1) * mask) / (th.sum(mask) + 1e-16)

        return binarization_mean, binariation_std, gestalt_mean

    def training_step(self, batch, batch_idx):
        results = self(*batch)

        loss = (
            results["mask_loss"]
            + results["rgb_loss"]
            + results["depth_loss"]
            + results["position_loss"]
            + results["gestalt_loss"]
            + results['z_loss']
        )

        binarization_mean, binariation_std, gestalt_mean = self.get_binarization_stats(results["gestalt"], 1)
        self.log("binarization_mean",  binarization_mean)
        self.log("binarization_std",   binariation_std)
        self.log("gestalt_mean",       gestalt_mean)

        self.log("loss",               loss, prog_bar=True)
        self.log("position_loss",      results["position_loss"])
        self.log("mask_loss",          results["mask_loss"])
        self.log("mean_iou",           results["mean_iou"])
        self.log("gestalt_loss",       results["gestalt_loss"])
        self.log("z_loss",             results["z_loss"])
        self.log("rgb_loss",           results["rgb_loss"])
        self.log("rgb_l1",             results["rgb_l1"])
        self.log("rgb_ssim",           results["rgb_ssim"])
        self.log("depth_loss",         results["depth_loss"])
        self.log("depth_l1",           results["depth_l1"])
        self.log("depth_ssim",         results["depth_ssim"])

        if self.num_updates < self.trainer.global_step:
            self.num_updates = self.trainer.global_step
            print("Epoch[{}|{}|{}|{:.2f}%]: {}, L: {:.2e}, P: {:.2e}, M: {:.2e}|{:.2f}%, z: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, depth: {:.2e}|{:.2e}|{:.2e}, bin: {:.2e}|{:.2e}|{:.3f}, t: {:.1f}%".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                str(self.timer),
                float(self.own_loggers['loss']),
                float(self.own_loggers['position_loss']),
                float(self.own_loggers['mask_loss']),
                float(self.own_loggers['mean_iou']) * 100,
                float(self.own_loggers['z_loss']),
                float(self.own_loggers['rgb_loss']),
                float(self.own_loggers['rgb_l1']),
                float(self.own_loggers['rgb_ssim']),
                float(self.own_loggers['depth_loss']),
                float(self.own_loggers['depth_l1']),
                float(self.own_loggers['depth_ssim']),
                float(self.own_loggers['binarization_mean']),
                float(self.own_loggers['binarization_std']),
                float(self.own_loggers['gestalt_mean']),
                results['time_weight'][0].item() + results['time_weight'][1].item() * 10 + results['time_weight'][2].item() * 100,
            ), flush=True)

        self.val_metrics = {}

        return loss

    def validation_step(self, batch, batch_idx):
        results = self(*batch)

        loss = (
            results["mask_loss"]
            + results["rgb_loss"]
            + results["depth_loss"]
            + results["position_loss"]
            + results["gestalt_loss"]
        )

        self.log("val_loss",               loss, prog_bar=True,      on_step=False, on_epoch=True)
        self.log("val_position_loss",      results["position_loss"], on_step=False, on_epoch=True)
        self.log("val_gestalt_loss",       results["gestalt_loss"],  on_step=False, on_epoch=True)
        self.log("val_mask_loss",          results["mask_loss"],     on_step=False, on_epoch=True)
        self.log("val_mean_iou",           results["mean_iou"],      on_step=False, on_epoch=True)
        self.log("val_z_loss",             results["z_loss"],        on_step=False, on_epoch=True)
        self.log("val_rgb_loss",           results["rgb_loss"],      on_step=False, on_epoch=True)
        self.log("val_rgb_l1",             results["rgb_l1"],        on_step=False, on_epoch=True)
        self.log("val_rgb_ssim",           results["rgb_ssim"],      on_step=False, on_epoch=True)
        self.log("val_depth_loss",         results["depth_loss"],    on_step=False, on_epoch=True)
        self.log("val_depth_l1",           results["depth_l1"],      on_step=False, on_epoch=True)
        self.log("val_depth_ssim",         results["depth_ssim"],    on_step=False, on_epoch=True)

        print("Test[{}|{}|{:.2f}%]: {}, Loss: {:.2e}, position: {:.2e}, mask: {:.2e}|{:.2f}%, z: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, depth: {:.2e}|{:.2e}|{:.2e}".format(
            self.trainer.local_rank,
            self.trainer.current_epoch,
            (batch_idx + 1) / len(self.trainer.val_dataloaders) * 100,
            str(self.timer),
            self.val_metrics['val_loss'] / (batch_idx + 1),
            self.val_metrics['val_position_loss'] / (batch_idx + 1),
            self.val_metrics['val_mask_loss'] / (batch_idx + 1),
            self.val_metrics['val_mean_iou'] / (batch_idx + 1) * 100,
            self.val_metrics['val_z_loss'] / (batch_idx + 1),
            self.val_metrics['val_rgb_loss'] / (batch_idx + 1),
            self.val_metrics['val_rgb_l1'] / (batch_idx + 1),
            self.val_metrics['val_rgb_ssim'] / (batch_idx + 1),
            self.val_metrics['val_depth_loss'] / (batch_idx + 1),
            self.val_metrics['val_depth_l1'] / (batch_idx + 1),
            self.val_metrics['val_depth_ssim'] / (batch_idx + 1),
        ), flush=True)

        return loss

    def configure_optimizers(self):
        return Ranger(self.net.parameters(), lr=self.lr, weight_decay=self.cfg.weight_decay)

