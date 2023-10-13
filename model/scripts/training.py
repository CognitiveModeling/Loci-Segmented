import os
import torch as th

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.lightning_objects import LociPretrainerDataModule
from data.lightning_proposal import LociProposalDataModule
from data.lightning_background import LociBackgroundDataModule
from data.lightning_loci import LociDataModule
from model.lightning.pretrainer import LociPretrainerModule
from model.lightning.proposal import LociProposalModule
from model.lightning.background import LociBackgroundModule
from model.lightning.loci import LociModule
from utils.configuration import Configuration
from utils.io import PeriodicCheckpoint
from model.pretrainer import LociPretrainer

def train_loci(cfg: Configuration, checkpoint_path, object_checkpoint_path, bg_checkpoint_path, proposal_checkpoint_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    state_dict = {}
    if object_checkpoint_path is not None and os.path.exists(object_checkpoint_path):
        object_model = LociPretrainerModule.load_from_checkpoint(object_checkpoint_path, cfg=cfg, map_location='cpu', strict=False)
        state_dict['encoder'] = object_model.net.encoder.state_dict()
        state_dict['decoder'] = object_model.net.decoder.state_dict()
        state_dict['gestalt_mean'] = object_model.net.gestalt_mean.data
        state_dict['gestalt_std']  = object_model.net.gestalt_std.data
        state_dict['std']   = object_model.net.std.data
        state_dict['depth'] = object_model.net.depth.data

    if bg_checkpoint_path is not None and os.path.exists(bg_checkpoint_path):
        bg_model = LociBackgroundModule.load_from_checkpoint(bg_checkpoint_path, cfg=cfg, strict=False)
        state_dict['background'] = bg_model.net.state_dict()

    data_module = LociDataModule(cfg)
    model       = LociModule(cfg, state_dict)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model = LociModule.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)

    if proposal_checkpoint_path is not None and os.path.exists(proposal_checkpoint_path):
        proposal_ckpt = th.load(proposal_checkpoint_path, map_location = 'cpu')
        state_dict = {}
        for k,v in proposal_ckpt['state_dict'].items():
            if k.startswith('net.'):
                state_dict[k.replace('net.', '')] = v
        
        model.net.proposal.load_state_dict(state_dict)

    # save initial model checkpints if loaded from object and background checkpoints
    if object_checkpoint_path is not None and bg_checkpoint_path is not None and checkpoint_path is None:
        th.save(model.state_dict(), f"out/{cfg.model_path}/Loci-pretrained.pt")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"out/{cfg.model_path}",
        filename="LociPretrainer-{epoch:02d}",
        save_top_k=-1,
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=1000,  # Save checkpoint every 3000 global steps
    )

    trainer = pl.Trainer(
        devices=([cfg.device] if cfg.single_gpu else -1),
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator='cuda',
        strategy="ddp_find_unused_parameters_true",
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
    )

    if cfg.validate:
        trainer.validate(model, data_module)
    else:
        trainer.fit(model, data_module)

def train_objects(cfg: Configuration, checkpoint_path, mask_path, depth_path, rgb_path, embeddings_path, stage1_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    state_dict = {}
    if mask_path is not None and os.path.exists(mask_path):
        state_dict['mask_pretrainer'] = {}
        state_dict['mask_decoder'] = {}
        model = th.load(mask_path, map_location='cpu')
        for k, v in model['state_dict'].items():
            if k.startswith('net.mask_pretrainer.'):
                state_dict['mask_pretrainer'][k.replace('net.mask_pretrainer.', '')] = v
            if k.startswith('net.mask_pretrainer.decoder'):
                state_dict['mask_decoder'][k.replace('net.mask_pretrainer.decoder.', '')] = v

    if depth_path is not None and os.path.exists(depth_path):
        state_dict['depth_pretrainer'] = {}
        state_dict['depth_decoder'] = {}
        model = th.load(depth_path, map_location='cpu')
        for k, v in model['state_dict'].items():
            if k.startswith('net.depth_pretrainer.'):
                state_dict['depth_pretrainer'][k.replace('net.depth_pretrainer.', '')] = v
            if k.startswith('net.depth_pretrainer.decoder'):
                state_dict['depth_decoder'][k.replace('net.depth_pretrainer.decoder.', '')] = v

    if rgb_path is not None and os.path.exists(rgb_path):
        state_dict['rgb_pretrainer'] = {}
        state_dict['rgb_decoder'] = {}
        model = th.load(rgb_path, map_location='cpu')
        for k, v in model['state_dict'].items():
            if k.startswith('net.rgb_pretrainer.decoder.') and '.f.' in k:
                state_dict['rgb_pretrainer'][k.replace('net.rgb_pretrainer.', '').replace('.f','')] = v
            elif k.startswith('net.rgb_pretrainer.'):
                state_dict['rgb_pretrainer'][k.replace('net.rgb_pretrainer.', '')] = v
            if k.startswith('net.rgb_pretrainer.decoder.') and '.f.' in k:
                state_dict['rgb_decoder'][k.replace('net.rgb_pretrainer.decoder.', '').replace('.f','')] = v
            elif k.startswith('net.rgb_pretrainer.decoder'):
                state_dict['rgb_decoder'][k.replace('net.rgb_pretrainer.decoder.', '')] = v

    data_module = LociPretrainerDataModule(cfg)
    model       = LociPretrainerModule(cfg, state_dict)

    if embeddings_path is not None and os.path.exists(embeddings_path):
        net = th.load(embeddings_path, map_location='cpu')
        for k, v in net['state_dict'].items():
            if k.startswith('net.mask_patch_embedding') or \
               k.startswith('net.depth_patch_embedding') or \
               k.startswith('net.rgb_patch_embedding') or \
               k.startswith('net.mask_patch_reconstruction') or \
               k.startswith('net.depth_patch_reconstruction') or \
               k.startswith('net.rgb_patch_reconstruction'):
                state_dict[k.replace('net.', '')] = v

        model.net.load_state_dict(state_dict, strict=False)
        model.net.copy_embeddings()

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model = LociPretrainerModule.load_from_checkpoint(checkpoint_path, cfg=cfg)

    if stage1_path is not None and os.path.exists(stage1_path):
        model = LociPretrainerModule.load_from_checkpoint(stage1_path, cfg=cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=f"out/{cfg.model_path}",
        filename="LociPretrainer-{epoch:02d}-{loss:.2f}",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=3000,  # Save checkpoint every 3000 global steps
    )

    callback_list = [checkpoint_callback, periodic_checkpoint_callback]

    trainer = pl.Trainer(
        devices=([cfg.device] if cfg.single_gpu else -1), 
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator='cuda',
        strategy="ddp_find_unused_parameters_true",
        max_epochs=cfg.epochs,
        callbacks=callback_list,
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
    )

    if cfg.validate:
        trainer.validate(model, data_module)
    else:
        trainer.fit(model, data_module)

def train_background(cfg: Configuration, checkpoint_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociBackgroundDataModule(cfg)
    model       = LociBackgroundModule(cfg)

    uncertaint_state_dict = {}
    for k, v in model.net.uncertainty_estimation.state_dict().items():
        uncertaint_state_dict[k] = v

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model = LociBackgroundModule.load_from_checkpoint(checkpoint_path, cfg=cfg)

    model.net.uncertainty_estimation.load_state_dict(uncertaint_state_dict) 

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"out/{cfg.model_path}",
        filename="LociBackground-{epoch:03d}",
        verbose=True,
        save_top_k=-1,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=3000,  # Save checkpoint every 3000 global steps
    )

    trainer = pl.Trainer(
        devices=([cfg.device] if cfg.single_gpu else -1),
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator='cuda',
        strategy="ddp",
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
    )

    if cfg.validate:
        trainer.validate(model, data_module)
    else:
        trainer.fit(model, data_module)

def train_proposal(cfg: Configuration, checkpoint_path, object_checkpoint_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    state_dict = None
    if object_checkpoint_path is not None and os.path.exists(object_checkpoint_path):
        object_model = LociPretrainerModule.load_from_checkpoint(object_checkpoint_path, cfg=cfg, strict=False)
        state_dict = object_model.net.mask_pretrainer.state_dict()

    data_module = LociProposalDataModule(cfg)
    model       = LociProposalModule(cfg, state_dict=state_dict)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model = LociProposalModule.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=f"out/{cfg.model_path}",
        filename="LociPretrainer-{epoch:02d}-{loss:.2f}",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=1000,  # Save checkpoint every 3000 global steps
    )

    callback_list = [checkpoint_callback, periodic_checkpoint_callback]

    trainer = pl.Trainer(
        devices=([cfg.device] if cfg.single_gpu else -1), 
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator='cuda',
        strategy="ddp",
        max_epochs=cfg.epochs,
        callbacks=callback_list,
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
    )

    trainer.fit(model, data_module)
