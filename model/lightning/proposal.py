import pytorch_lightning as pl
import torch as th
import numpy as np
from torch.utils.data import DataLoader
from utils.optimizers import Ranger
from utils.configuration import Configuration
from nn.proposal import LociProposal
from utils.io import UEMA, Timer
import torch.distributed as dist
from einops import rearrange, repeat, reduce

class LociProposalModule(pl.LightningModule):
    def __init__(self, cfg: Configuration, state_dict=None):
        super().__init__()
        self.cfg = cfg

        print(f"RANDOM SEED: {cfg.seed}")
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        input_size = cfg.model.input_size
        if not isinstance(input_size, tuple) and not isinstance(input_size, list):
            input_size = (input_size, input_size)

        self.net = LociProposal(
            size           = input_size,
            num_slots      = cfg.model.position_proposal.num_slots,
            encoder_blocks = cfg.model.position_proposal.encoder_blocks,
            decoder_blocks = cfg.model.position_proposal.decoder_blocks,
            base_channels  = cfg.model.position_proposal.base_channels,
            depth_input    = cfg.model.input_depth,
        )

        if state_dict is not None:
            self.net.mask_pretrainer.load_state_dict(state_dict)
            print("Loaded mask pretrainer")

        self.lr = self.cfg.learning_rate
        self.own_loggers = {}
        self.timer = Timer()
        self.val_metrics = {}

        self.num_updates = -1

    def forward(self, input_rgb, input_depth, input_instance_masks):
        if self.cfg.model.input_depth:
            return self.net(input_instance_masks, input_depth)
        
        return self.net(input_instance_masks, input_depth, input_rgb)

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

    def training_step(self, batch, batch_idx):
        results = self(*batch)

        loss = results["regularizer_loss"] + results['mask_loss']

        self.log("loss",                      loss, prog_bar=True)
        self.log("regularizer_loss",          results["regularizer_loss"])
        self.log("iou",                       results["iou"])
        self.log("mask_loss",                 results["mask_loss"])

        if self.num_updates < self.trainer.global_step:
            self.num_updates = self.trainer.global_step
            print("Epoch[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, mask-loss: {:.2e}, reg-loss: {:.2e}, IoU: {:.2f}%".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                str(self.timer),
                float(self.own_loggers['loss']),
                float(self.own_loggers['mask_loss']),
                float(self.own_loggers['regularizer_loss']),
                float(self.own_loggers['iou']),
            ), flush=True)

        self.val_metrics = {}

        return loss

    def validation_step(self, batch, batch_idx):
        results = self(*batch)

        loss = results["regularizer_loss"] + results['mask_loss']

        self.log("val_loss",               loss, prog_bar=True,         on_step=False, on_epoch=True)
        self.log("val_regularizer_loss",   results["regularizer_loss"],    on_step=False, on_epoch=True)
        self.log("val_iou",                results["iou"],     on_step=False, on_epoch=True)
        self.log("val_mask_loss",          results["mask_loss"],        on_step=False, on_epoch=True)

        print("Test[{}|{}|{:.2f}%]: {}, Loss: {:.2e}, mask-loss: {:.2e}, reg-loss: {:.2e}, IoU: {:.2f}".format(
            self.trainer.local_rank,
            self.trainer.current_epoch,
            (batch_idx + 1) / len(self.trainer.val_dataloaders) * 100,
            str(self.timer),
            self.val_metrics['val_loss'] / (batch_idx + 1),
            self.val_metrics['val_mask_loss'] / (batch_idx + 1),
            self.val_metrics['val_regularizer_loss'] / (batch_idx + 1),
            self.val_metrics['val_iou'] / (batch_idx + 1),
        ), flush=True)

        return loss

    def configure_optimizers(self):
        return Ranger(self.net.parameters(), lr=self.lr, weight_decay=self.cfg.weight_decay)

