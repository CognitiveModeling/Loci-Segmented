import pytorch_lightning as pl
import torch as th
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils.io import UEMA, Timer
from utils.optimizers import Ranger
from nn.background import UncertantyBackground
from utils.loss import MaskedYCbCrL2SSIMLoss, UncertaintyYCbCrL2SSIMLoss, SSIM, GestaltLoss

class LociBackgroundModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.own_loggers = {}
        self.timer = Timer()
        self.val_metrics = {}

        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        self.net = UncertantyBackground(
            uncertainty_threshold   = self.cfg.model.background.uncertainty_threshold,
            motion_context_size     = self.cfg.model.background.motion_context_size,
            depth_context_size      = self.cfg.model.background.depth_context_size,
            latent_channels         = self.cfg.model.background.latent_channels,
            num_layers              = self.cfg.model.background.num_layers,
            depth_input             = self.cfg.model.input_depth,
        )

        self.rgb_loss = MaskedYCbCrL2SSIMLoss()
        self.maskloss = nn.MSELoss()
        self.uncertainty_rgb_loss = UncertaintyYCbCrL2SSIMLoss()
        self.ssim = SSIM()
        self.context_loss = GestaltLoss()

        self.last_input = None
        self.last_rgb = None
        self.last_depth = None
        self.last_fg_mask = None
        self.last_uncertainty = None

        self.num_updates = -1

    def log(self, name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True):
        super().log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

        if name.startswith("val_"):
            if name not in self.val_metrics:
                self.val_metrics[name] = 0
                print("Adding metric: ", name)

            self.val_metrics[name] += value.item() if isinstance(value, th.Tensor) else value
        else:
            if name not in self.own_loggers:
                self.own_loggers[name] = UEMA(10000)

            self.own_loggers[name].update(value.item() if isinstance(value, th.Tensor) else value)

    def norm(sel, x):
        return th.sigmoid((x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + 1e-6))

    def get_binarization_stats(self, context):

        binarization      = th.min(th.abs(context), th.abs(1 - context)).detach()
        binarization_mean = th.mean(binarization)
        binariation_std   = th.std(binarization)
        context_mean      = th.mean(context)

        return binarization_mean, binariation_std, context_mean

    def forward(self, batch, batch_idx, prefix):
        (
            source_rgb, 
            source_depth, 
            source_fg_mask, 
            target_rgb, 
            target_depth, 
            target_fg_mask, 
            use_depth, 
            use_fg_masks, 
            delta_t, 
            uncertainty_regularizer, 
            depth_weighting, 
            rgb_loss_factor,
            rgb_warmup,
            depth_warmup,
            color_input,
        ) = batch

        # reshape for easy broadcasting
        use_depth               = use_depth.view(-1, 1, 1, 1).float()
        use_fg_masks            = use_fg_masks.view(-1, 1, 1, 1).float() 
        uncertainty_regularizer = uncertainty_regularizer.view(-1, 1, 1, 1).float()
        depth_weighting         = depth_weighting.view(-1, 1, 1, 1).float()
        rgb_loss_factor         = rgb_loss_factor.view(-1, 1, 1, 1).float() * use_depth + (1 - use_depth)
        rgb_warmup              = (rgb_warmup.view(-1, 1, 1, 1) > self.trainer.global_step).float()
        depth_warmup            = (depth_warmup.view(-1, 1, 1, 1) > self.trainer.global_step).float()
        color_input             = color_input.view(-1, 1, 1, 1).float()
        warmup                  = th.maximum(rgb_warmup, depth_warmup)


        source_confidence = (source_fg_mask > 0.5).float() * (source_fg_mask - 0.5) * 2 + (source_fg_mask <= 0.5).float() * (0.5 - source_fg_mask) * 2
        target_confidence = (target_fg_mask > 0.5).float() * (target_fg_mask - 0.5) * 2 + (target_fg_mask <= 0.5).float() * (0.5 - target_fg_mask) * 2
        depth_weight      = th.clip(target_depth, 0.1, 1) * use_depth * depth_weighting + use_depth * (1 - depth_weighting)
        depth_mask        = (target_depth >= 0).float()

        source = th.cat([source_rgb, source_depth], dim=1) if self.cfg.model.input_depth else source_rgb
        target = th.cat([target_rgb, target_depth], dim=1) if self.cfg.model.input_depth else target_rgb

        # run uncertainty estimation
        source_uncertainty, source_uncertainty_noised = self.net.uncertainty_estimation(source)
        target_uncertainty, target_uncertainty_noised = self.net.uncertainty_estimation(target)

        supervised_uncertainty_loss = (
            self.maskloss(source_uncertainty, (source_fg_mask > 0.5).float()) + 
            self.maskloss(target_uncertainty, (target_fg_mask > 0.5).float())
        ) * self.cfg.model.background.supervision_factor

        output_rgb, output_depth, motion_context, depth_context = self.net(
            source, 
            target, 
            source_uncertainty.detach() * (1 - warmup) + warmup * th.zeros_like(source_uncertainty.detach()),
            target_uncertainty.detach() * (1 - warmup) + warmup * th.zeros_like(target_uncertainty.detach()),
            delta_t,
            color_input,
        )

        # only apply loss to certain regions
        prediction_mask = (target_uncertainty < self.cfg.model.background.uncertainty_threshold).float().detach()

        # only applay loss to foreground regions
        prediction_mask = prediction_mask * (1 - use_fg_masks) + use_fg_masks * (target_fg_mask < self.cfg.model.background.uncertainty_threshold).float().detach()

        # during warmup apply loss to all regions
        prediction_mask = prediction_mask * (1 - warmup) + warmup

        unsupervised_uncertainty_loss = ( 
            uncertainty_regularizer * th.mean(th.abs(target_uncertainty_noised), dim=(1,2,3), keepdim=True) +
            th.mean(depth_weight * th.abs(output_depth - target_depth).detach() * (1 - target_uncertainty_noised), dim=(1,2,3), keepdim=True) +
            rgb_loss_factor * th.mean(th.abs(output_rgb - target_rgb).detach() * (1 - target_uncertainty_noised), dim=(1,2,3), keepdim=True)
        )

        uncertainty_loss = th.mean(supervised_uncertainty_loss * use_fg_masks + unsupervised_uncertainty_loss * (1 - use_fg_masks))

        depth_loss = rgb_loss = 0

        depth_loss         = th.mean(th.abs(output_depth - target_depth) * prediction_mask * depth_mask)
        rgb_loss, l2, ssim = self.rgb_loss(output_rgb, target_rgb, prediction_mask * (1 - depth_warmup))

        self.log(f'{prefix}_rgb_l2',   l2.item())
        self.log(f'{prefix}_rgb_ssim', ssim.item())

        loss = uncertainty_loss + depth_loss + rgb_loss + self.context_loss(depth_context) 

        depth_context_bin_mean, depth_context_bin_std, depth_context_mean = self.get_binarization_stats(depth_context)

        self.log(f'{prefix}_rgb_loss',                      rgb_loss.item())
        self.log(f'{prefix}_depth_loss',                    depth_loss.item())
        self.log(f'{prefix}_loss',                          loss.item())
        self.log(f'{prefix}_uncertainty',                   th.mean(target_uncertainty).item())
        self.log(f'{prefix}_prediction_mask',               th.mean(prediction_mask).item())
        self.log(f'{prefix}_supervised_uncertainty_loss',   th.mean(supervised_uncertainty_loss * use_fg_masks).item())
        self.log(f'{prefix}_unsupervised_uncertainty_loss', th.mean(unsupervised_uncertainty_loss * (1 - use_fg_masks)).item())
        self.log(f'{prefix}_depth_context_bin_mean',        depth_context_bin_mean.item())
        self.log(f'{prefix}_depth_context_bin_std',         depth_context_bin_std.item())
        self.log(f'{prefix}_depth_context_mean',            depth_context_mean.item())

        return loss


    def training_step(self, batch, batch_idx):

        loss = self(batch, batch_idx, "train")
        
        if self.num_updates < self.trainer.global_step:
            self.num_updates = self.trainer.global_step
            print("Epoch[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, U: {:.2e}|{:.2e}|{:.2e}, M: {:.2e}, depth: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, D-bin: {:.2e}|{:.2e}|{:.3f}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                str(self.timer),
                float(self.own_loggers['train_loss']),
                float(self.own_loggers['train_supervised_uncertainty_loss']),
                float(self.own_loggers['train_unsupervised_uncertainty_loss']),
                float(self.own_loggers['train_uncertainty']),
                float(self.own_loggers['train_prediction_mask']),
                float(self.own_loggers['train_depth_loss']),
                float(self.own_loggers['train_rgb_loss']),
                float(self.own_loggers['train_rgb_l2']),
                float(self.own_loggers['train_rgb_ssim']),
                float(self.own_loggers['train_depth_context_bin_mean']),
                float(self.own_loggers['train_depth_context_bin_std']),
                float(self.own_loggers['train_depth_context_mean']),
            ), flush=True)

        self.val_metrics = {}

        return loss

    def validation_step(self, batch, batch_idx):
        self(batch, batch_idx, "val")
        
        if self.num_updates < self.trainer.global_step:
            self.num_updates = self.trainer.global_step
            print("Test[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, U: {:.2e}|{:.2e}|{:.2e}, M: {:.2e}, depth: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.val_dataloaders[0]) * 100,
                str(self.timer),
                self.val_metrics['val_loss'] / (batch_idx + 1),
                self.val_metrics['val_supervised_uncertainty_loss'] / (batch_idx + 1),
                self.val_metrics['val_unsupervised_uncertainty_loss'] / (batch_idx + 1),
                self.val_metrics['val_uncertainty'] / (batch_idx + 1),
                self.val_metrics['val_prediction_mask'] / (batch_idx + 1),
                self.val_metrics['val_depth_loss'] / (batch_idx + 1),
                self.val_metrics['val_rgb_loss'] / (batch_idx + 1),
                self.val_metrics['val_rgb_l2'] / (batch_idx + 1),
                self.val_metrics['val_rgb_ssim'] / (batch_idx + 1)
            ), flush=True)

    def test_step(self, batch, batch_idx):
        # Optional: Implement the test step
        pass

    def configure_optimizers(self):
        optimizer = Ranger([
            {'params': self.net.parameters(), 'lr': self.cfg.learning_rate, "weight_decay": self.cfg.weight_decay},
        ])
        return optimizer

