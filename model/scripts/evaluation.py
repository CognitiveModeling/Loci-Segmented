import torch as th
from torch.utils.data import Dataset, DataLoader
from torch import nn

import h5py
import os
from utils.configuration import Configuration
from model.loci import Loci
from utils.utils import LambdaModule, Gaus2D, Prioritize, MaskCenter
from utils.io import Timer
import numpy as np
import cv2
from pathlib import Path
import shutil
import pickle
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from model.pretrainer import LociPretrainer
import pytorch_lightning as pl
from data.lightning_objects import LociPretrainerDataModule
from data.lightning_background import LociBackgroundDataModule
from data.lightning_loci import LociDataModule
from data.lightning_proposal import LociProposalDataModule
from model.lightning.pretrainer import LociPretrainerModule
from model.lightning.background import LociBackgroundModule
from model.lightning.loci import LociModule
from model.lightning.proposal import LociProposalModule
from utils.loss import SSIM

#def preprocess(tensor, scale=1, normalize=False, mean_std_normalize=False, size=None, add_text=False, text="", position=(10, 30), font_scale=1, font_color=(255,255,255), outline_color=(0,0,0), font_thickness=2):
def preprocess(tensor, scale=1, normalize=False, mean_std_normalize=False, size=None, add_text=False, text="", position=(10, 30), font_scale=0.7, font_color=(255,255,255), outline_color=(0,0,0), font_thickness=1, interpolation_mode='bicubic'):

    if normalize:
        min_ = th.min(tensor)
        max_ = th.max(tensor)
        tensor = (tensor - min_) / (max_ - min_)

    if mean_std_normalize:
        mean = th.mean(tensor)
        std = th.std(tensor)
        tensor = th.clip((tensor - mean) / (2 * std), -1, 1) * 0.5 + 0.5

    if scale > 1:
        tensor = F.interpolate(tensor, scale_factor=scale, mode=interpolation_mode, align_corners=True)

    if size is not None:
        tensor = F.interpolate(tensor, size=size, mode=interpolation_mode, align_corners=True)

    if add_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = (tensor[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        img = cv2.UMat(img).get()
        img = cv2.putText(img, text, position, font, font_scale, outline_color, font_thickness+2, cv2.LINE_AA)
        img = cv2.putText(img, text, position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        tensor = th.tensor(img.transpose(2,0,1), device=tensor.device).unsqueeze(0) / 255.0

    return tensor

def color_mask(mask):

    colors = th.tensor([
	[ 255,   0,   0 ],
	[   0,   0, 255 ],
	[ 255, 255,   0 ],
	[ 255,   0, 255 ],
	[   0, 255, 255 ],
	[   0, 255,   0 ],
	[ 255, 128,   0 ],
	[ 128, 255,   0 ],
	[ 128,   0, 255 ],
	[ 255,   0, 128 ],
	[   0, 255, 128 ],
	[   0, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 255 ],
	[ 128, 255, 255 ],
	[ 128, 255, 255 ],
	[ 255, 255, 128 ],
	[ 255, 255, 128 ],
	[ 255, 128, 255 ],
	[ 128,   0,   0 ],
	[   0,   0, 128 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
    ], device = mask.device) / 255.0

    colors = colors.view(1, -1, 3, 1, 1)
    mask = mask.unsqueeze(dim=2)

    return th.sum(colors[:,:mask.shape[1]] * mask, dim=1)


def priority_to_img(priority, h, w):

    imgs = []

    for p in range(priority.shape[2]):

        img = np.zeros((h,w,3), np.uint8)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        text_position          = (h // 6, w //2)
        font_scale             = w / 256
        font_color             = (255,255,255)
        thickness              = 2
        lineType               = 2

        cv2.putText(img,f'{priority[0,0,p].item():.2e}',
            text_position,
            font,
            font_scale,
            font_color,
            thickness,
            lineType)

        imgs.append(rearrange(th.tensor(img, device=priority.device), 'h w c -> 1 1 c h w'))

    return imgs

def to_rgb_object(tensor, o):
    colors = th.tensor([
	[ 255,   0,   0 ],
	[   0,   0, 255 ],
	[ 255, 255,   0 ],
	[ 255,   0, 255 ],
	[   0, 255, 255 ],
	[   0, 255,   0 ],
	[ 255, 128,   0 ],
	[ 128, 255,   0 ],
	[ 128,   0, 255 ],
	[ 255,   0, 128 ],
	[   0, 255, 128 ],
	[   0, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 255 ],
	[ 128, 255, 255 ],
	[ 128, 255, 255 ],
	[ 255, 255, 128 ],
	[ 255, 255, 128 ],
	[ 255, 128, 255 ],
	[ 128,   0,   0 ],
	[   0,   0, 128 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
    ], device = tensor.device) / 255.0

    colors = colors.view(48,3,1,1)
    return colors[o] * tensor

def to_rgb(tensor: th.Tensor):
    return th.cat((
        tensor * 0.6 + 0.4,
        tensor, 
        tensor
    ), dim=1)


def save(cfg: Configuration, dataset: Dataset, checkpoint_path, proposal_checkpoint_path, size, object_view, nice_view, individual_views, add_text):

    np.random.seed(1234)
    th.manual_seed(1234)

    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    data_module = LociDataModule(cfg)
    model       = LociModule(cfg)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):

        # ends with ckpt
        if checkpoint_path[-4:] == 'ckpt':
            model = LociModule.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
        else:
            model.load_state_dict(th.load(checkpoint_path, map_location=device))

    if proposal_checkpoint_path != "" and os.path.exists(proposal_checkpoint_path):
        proposal_ckpt = th.load(proposal_checkpoint_path, map_location = 'cpu')
        state_dict = {}
        for k,v in proposal_ckpt['state_dict'].items():
            if k.startswith('net.'):
                state_dict[k.replace('net.', '')] = v
        
        model.net.proposal.load_state_dict(state_dict)

    net = model.net.to(device=device)
    net.eval()

    dataloader = data_module.val_dataloader()
    
    gaus2d = Gaus2D(size, position_limit=3.5).to(device)
    mseloss = nn.MSELoss()

    prioritize = Prioritize(cfg_net.num_slots).to(device)

    last_input = None
    last_rgb = None
    last_depth = None
    last_time_steps = None
    output = None

    i = -2
    teacher_forcing = ((cfg.teacher_forcing // cfg.backprop_steps) * cfg.backprop_steps + 1) if cfg.sequence_len > 1 else cfg.teacher_forcing
    latent_size = [cfg_net.input_size[0] // 16, cfg_net.input_size[1] // 16]

    with th.no_grad():
        for input in dataloader:

            tensor_rgb        = input[0].to(device)
            tensor_depth      = input[1].to(device) 
            time_steps        = input[2].to(device)
            use_depth         = input[3].to(device)

            timestep = time_steps[0,0].item()

            if timestep == -teacher_forcing:
                i += 1
                net.reset_state()

                output = { 
                    'reconstruction' : {
                        'rgb': None,
                        'depth_raw': None,
                        'mask': None,
                        'mask_raw': None,
                        'occlusion': None,
                        'position': None,
                        'gestalt': None,
                        'priority': None,
                        'output_depth': None,
                    },
                    'prediction' : {
                        'bg_rgb': None,
                        'bg_depth': None,
                        'output_depth': None,
                        'rgb': None,
                        'depth_raw': None,
                        'mask': None,
                        'mask_raw': None,
                        'occlusion': None,
                        'position': None,
                        'gestalt': None,
                        'priority': None,
                    }
                }

                if cfg.model.inference_mode == "segmentation":
                    bg_tensor        = th.cat((tensor_rgb[:,0], tensor_depth[:,0]), dim=1) if cfg.model.input_depth else tensor_rgb[:,0]
                    uncertainty_cur = net.background.uncertainty_estimation(bg_tensor)[0]
                    fg_mask         = (uncertainty_cur > 0.8).float()
                    
                    B, _, _, H, W = tensor_rgb.shape
                    gt_masks = th.zeros((B, cfg.model.position_proposal.num_slots, H, W), device = tensor_rgb.device)
                
                    results = net.proposal(gt_masks, tensor_depth[:,0], tensor_rgb[:,0], fg_mask = fg_mask) 
                
                    seg_position = results['position']
                    seg_mask     = results['mask']
                    
                    # sort by mask size
                    seg_mask_sum = reduce(seg_mask, 'b o h w -> b o', 'sum')
                    sorted_values, sorted_indices = th.sort(seg_mask_sum, dim=1, descending=True)
                
                    # Using advanced indexing to sort the masks and positions
                    sorted_seg_mask     = seg_mask[th.arange(seg_mask.size(0)).unsqueeze(1), sorted_indices]
                    sorted_seg_position = seg_position[th.arange(seg_position.size(0)).unsqueeze(1), sorted_indices]
                
                    sorted_seg_position = sorted_seg_position[:,:cfg.model.num_slots]
                    sorted_seg_mask     = sorted_seg_mask[:,:cfg.model.num_slots]
                
                    output['reconstruction']['position'] = rearrange(sorted_seg_position, 'b n c -> b (n c)')
                    output['reconstruction']['mask']     = th.cat((sorted_seg_mask, 1 - reduce(results['mask'], 'b n h w -> b 1 h w', 'max')), dim=1)
            else:
                tensor_rgb   = th.cat((last_rgb, tensor_rgb), dim=1)
                tensor_depth = th.cat((last_depth, tensor_depth), dim=1)
                time_steps   = th.cat((last_time_step, time_steps), dim=1)

            last_rgb       = tensor_rgb[:, -1:]
            last_depth     = tensor_depth[:, -1:]
            last_time_step = time_steps[:, -1:]

            for time_step in range(len(time_steps[0])-1):
                timestep = time_steps[0,time_step].item()
                t = time_step if time_steps[0,time_step].item() >= 0 else 0

                input_rgb   = tensor_rgb[:, t]
                input_depth = tensor_depth[:, t]

                target_rgb   = tensor_rgb[:, t+1 if timestep >= 0 else 0]
                target_depth = tensor_depth[:, t+1 if timestep >= 0 else 0]

                output_last = output['prediction'] if time_steps[0,t].item() > 0 else output['reconstruction']
                output = net(
                    input_rgb       = input_rgb,
                    input_depth     = input_depth if cfg.model.input_depth else output_last['output_depth'],
                    bg_rgb_last     = output['prediction']['bg_rgb'],
                    bg_depth_last   = output['prediction']['bg_depth'],
                    rgb_last        = output_last['rgb'],
                    depth_raw_last  = output_last['depth_raw'],
                    mask_last       = output_last['mask'],
                    mask_raw_last   = output_last['mask_raw'],
                    occlusion_last  = output_last['occlusion'],
                    position_last   = output_last['position'],
                    gestalt_last    = output_last['gestalt'],
                    priority_last   = output_last['priority'],
                    teacher_forcing = time_steps[0,t].item() < 0,
                    reset           = False,
                    detach          = False,
                    evaluate        = True, 
                    test            = False,
                )

                output_next = output['prediction'] if time_steps[0,t].item() >= 0 else output['reconstruction']

                bg_depth_next        = output['prediction']['bg_depth']
                background_next      = output['prediction']['bg_rgb']
                output_rgb_next      = output_next['output_rgb']
                output_depth_next    = output_next['output_depth']
                mask_next            = output_next['mask']
                rgb_next             = output_next['rgb']
                depth_next           = output_next['depth']
                position_next        = output_next['position']
                gestalt_next         = output_next['gestalt']
                priority_next        = output_next['priority']
                uncertainty_cur      = output['reconstruction']['uncertainty']

                if timestep == 5:
                    _gestalt  = rearrange(gestalt_next,  'b (n c) -> b n c', n = cfg_net.num_slots)
                    _position = rearrange(position_next, 'b (n c) -> b n c', n = cfg_net.num_slots)
                    for n in range(0, cfg_net.num_slots):
                        with open(f'latent-{i:04d}-{n:02d}.pickle', "wb") as outfile:
                            state = {
                                "gestalt":  th.round(th.clip(_gestalt[0:1,n], 0, 1)),
                                "position": _position[0:1,n],
                            }
                            pickle.dump(state, outfile)

                print(f'Saving[{timestep+teacher_forcing:3d}/{i+1}/{len(dataloader)}]: {(i*100) / len(dataloader):.3f}%')

                highlited_target_rgb = target_rgb
                if mask_next is not None:
                    grayscale                 = target_rgb[:,0:1] * 0.299 + target_rgb[:,1:2] * 0.587 + target_rgb[:,2:3] * 0.114
                    object_mask_next          = th.sum(mask_next[:,:-1], dim=1).unsqueeze(dim=1)
                    highlited_target_rgb  = grayscale * (1 - object_mask_next) 
                    highlited_target_rgb += grayscale * object_mask_next * 0.3333333 
                    highlited_target_rgb  = highlited_target_rgb + color_mask(mask_next[:,:-1]) * 0.6666666

                highlited_target_depth = target_depth if target_depth is not None else th.zeros_like(target_rgb)
                if mask_next is not None:
                    grayscale                 = target_depth if target_depth is not None else th.zeros_like(target_rgb)
                    object_mask_next          = th.sum(mask_next[:,:-1], dim=1).unsqueeze(dim=1)
                    highlited_target_depth  = grayscale * (1 - object_mask_next) 
                    highlited_target_depth += grayscale * object_mask_next * 0.3333333 
                    highlited_target_depth  = highlited_target_depth + color_mask(mask_next[:,:-1]) * 0.6666666

                xy_next         = rearrange(position_next, 'b (o c) -> (b o) c', o = cfg_net.num_slots)[:,:2]
                position_next   = th.cat((position_next, th.zeros_like(position_next[:,:3]), th.ones_like(position_next[:,:1])), dim=1)
                position_next2d = rearrange(position_next, 'b (o c) -> (b o) c', o=cfg_net.num_slots+1)
                position_next2d = th.cat((position_next2d[:,:3], th.clamp(position_next2d[:,3:], 1 / min(latent_size), 1)), dim=1)
                position_next2d = gaus2d(position_next2d)
                position_next2d = rearrange(position_next2d, '(b o) 1 h w -> b o 1 h w', o=cfg_net.num_slots+1)
                                 
                rgb_next = th.cat((rgb_next, background_next), dim=1)
                rgb_next = rearrange(rgb_next, 'b (o c) h w -> b o c h w', c = 3)

                depth_next = th.cat((depth_next, bg_depth_next), dim=1)
                depth_next = rearrange(depth_next, 'b (o 1) h w -> b o 1 h w')
                mask_next  = rearrange(mask_next, 'b (o 1) h w -> b o 1 h w')

                object_mask_next = reduce(output_next['mask'][:,:-1], 'b c h w -> b 1 h w', 'sum')

                output_dir = 'individual_images'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if object_view or individual_views:
                    num_slots     = cfg_net.num_slots + 1
                    object_width  = int(np.ceil(((size[1] * 3 + 18) - (num_slots - 1) * 6) / num_slots))
                    object_height = int(np.ceil((object_width / size[1]) * size[0]))
                    padding = (object_width * num_slots + (num_slots - 1) * 6) - size[1] * 3
                    obj_size = (object_height, object_width)
                    
                    width  = size[1] * 3 + 18*3 + padding
                    height = size[0] * 2 + 18*4 + 6*3 + object_height*4

                    img = th.ones((3, height, width), device = rgb_next.device) * 0.2

                    # Process the main images
                    input_rgb_img = preprocess(input_rgb, size=size, add_text=add_text, text=f"GT RGB t = {timestep}")[0]
                    input_depth_img = preprocess(input_depth if use_depth[0].item() == 1 else th.zeros_like(input_rgb), size=size, add_text=add_text, text=f"GT Depth t = {timestep}")[0]
                    highlited_target_rgb_img = preprocess(highlited_target_rgb, size=size, add_text=add_text, text=f"Segmentation t = {timestep+1}")[0]
                    output_rgb_img = preprocess(output_rgb_next, size=size, add_text=add_text, text=f"RGB Output t = {timestep+1}")[0]
                    output_depth_img = preprocess(output_depth_next, size=size, add_text=add_text, text=f"Depth Output t = {timestep+1}")[0]
                    uncertainty_img = preprocess(uncertainty_cur, size=size, add_text=add_text, text=f"Uncertainty t = {timestep}", interpolation_mode='bilinear')[0]
    
                    # Save the main images individually
                    if individual_views:
                        cv2.imwrite(os.path.join(output_dir, f'input_rgb-{i:04d}-{timestep+teacher_forcing:03d}.jpg'), input_rgb_img.cpu().numpy().transpose(1,2,0) * 255)
                        cv2.imwrite(os.path.join(output_dir, f'input_depth-{i:04d}-{timestep+teacher_forcing:03d}.jpg'), input_depth_img.cpu().numpy().transpose(1,2,0) * 255)
                        cv2.imwrite(os.path.join(output_dir, f'highlited_target_rgb-{i:04d}-{timestep+teacher_forcing:03d}.jpg'), highlited_target_rgb_img.cpu().numpy().transpose(1,2,0) * 255)
                        cv2.imwrite(os.path.join(output_dir, f'output_rgb-{i:04d}-{timestep+teacher_forcing:03d}.jpg'), output_rgb_img.cpu().numpy().transpose(1,2,0) * 255)
                        cv2.imwrite(os.path.join(output_dir, f'output_depth-{i:04d}-{timestep+teacher_forcing:03d}.jpg'), output_depth_img.cpu().numpy().transpose(1,2,0) * 255)
                        cv2.imwrite(os.path.join(output_dir, f'uncertainty-{i:04d}-{timestep+teacher_forcing:03d}.jpg'), uncertainty_img.cpu().numpy().transpose(1,2,0) * 255)
    
                    # Add the images to the main img as before
                    img[:, 18:size[0]+18, 18:size[1]+18] = input_rgb_img
                    img[:, 18:size[0]+18, 18*2+size[1]:18*2+size[1]*2] = input_depth_img
                    img[:, 18:size[0]+18, 18*3+size[1]*2:18*3+size[1]*3] = highlited_target_rgb_img
                    img[:, -size[0]-18:-18, 18:size[1]+18] = output_rgb_img
                    img[:, -size[0]-18:-18, 18*2+size[1]:18*2+size[1]*2] = output_depth_img
                    img[:, -size[0]-18:-18, 18*3+size[1]*2:18*3+size[1]*3] = uncertainty_img

                    for o in range(num_slots):
                        prefix = f"Slot {o}" if o < num_slots - 1 else "Background"
                        # Process the images
                        rgb_img = preprocess(rgb_next[:,o], add_text=add_text, text=f"{prefix} RGB", font_scale=0.7, position=(10,30))
                        depth_img = preprocess(depth_next[:,o], add_text=add_text, text=f"{prefix} Depth", font_scale=0.7, position=(10,30))
                        mask_img = preprocess(to_rgb_object(mask_next[:,o], o), add_text=add_text, text=f"{prefix} Mask", font_scale=0.7, position=(10,30))
                        position_img = preprocess(to_rgb_object(position_next2d[:,o], o), add_text=add_text, text=f"{prefix} Position", font_scale=0.7, position=(10,30))
    
                        # Save the individual images
                        if individual_views:
                            cv2.imwrite(os.path.join(output_dir, f'rgb-{i:04d}-{timestep+teacher_forcing:03d}-obj{o:02d}.jpg'), rgb_img[0].cpu().numpy().transpose(1,2,0) * 255)
                            cv2.imwrite(os.path.join(output_dir, f'depth-{i:04d}-{timestep+teacher_forcing:03d}-obj{o:02d}.jpg'), depth_img[0].cpu().numpy().transpose(1,2,0) * 255)
                            cv2.imwrite(os.path.join(output_dir, f'mask-{i:04d}-{timestep+teacher_forcing:03d}-obj{o:02d}.jpg'), mask_img[0].cpu().numpy().transpose(1,2,0) * 255)
                            cv2.imwrite(os.path.join(output_dir, f'position-{i:04d}-{timestep+teacher_forcing:03d}-obj{o:02d}.jpg'), position_img[0].cpu().numpy().transpose(1,2,0) * 255)

                        # resize the images
                        rgb_img = preprocess(rgb_img, size=obj_size)[0]
                        depth_img = preprocess(depth_img, size=obj_size)[0]
                        mask_img = preprocess(mask_img, size=obj_size)[0]
                        position_img = preprocess(position_img, size=obj_size)[0]
    
                        # Add the images to the main img as before
                        img[:,size[0]+18*2:size[0]+18*2+object_height,18+(object_width+6)*o:18+(object_width+6)*o+object_width] = rgb_img
                        img[:,size[0]+18*2+6+object_height:size[0]+18*2+6+2*object_height,18+(object_width+6)*o:18+(object_width+6)*o+object_width] = depth_img
                        img[:,size[0]+18*2+2*(6+object_height):size[0]+18*2+2*6+3*object_height,18+(object_width+6)*o:18+(object_width+6)*o+object_width] = mask_img
                        img[:,size[0]+18*2+3*(6+object_height):size[0]+18*2+3*6+4*object_height,18+(object_width+6)*o:18+(object_width+6)*o+object_width] = position_img

                    img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
                    cv2.imwrite(f'loci-grid-{i:04d}-{timestep+teacher_forcing:03d}.jpg', img)



def save_bg(cfg: Configuration, dataset: Dataset, file, size, add_text, individual_views):

    np.random.seed(1234)
    th.manual_seed(1234)

    #assert(cfg.sequence_len == 2)
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociBackgroundDataModule(cfg)
    dataloader  = data_module.val_dataloader()

    if file != '':
        model = LociBackgroundModule.load_from_checkpoint(file, cfg=cfg, strict=False).to(device)
    else:
        model = LociBackgroundModule(cfg).to(device)

    # create model 
    net = model.net
    net.eval()

    ssim = SSIM()
    
    with th.no_grad():
        for i, batch in enumerate(dataloader):

            batch = [b.to(device) for b in batch]

            source_rgb, source_depth, source_fg_mask, target_rgb, target_depth, target_fg_mask, use_depth, use_fg_masks, delta_t, _, _, _, _, _, input_mode = batch

            source = th.cat([source_rgb, source_depth], dim=1) if cfg.model.input_depth else source_rgb
            target = th.cat([target_rgb, target_depth], dim=1) if cfg.model.input_depth else target_rgb

            source_uncertainty, source_uncertainty_noised = net.uncertainty_estimation(source)
            target_uncertainty, target_uncertainty_noised = net.uncertainty_estimation(target)

            output_rgb, output_depth, motion_context, depth_context = net(
                #source_rgb, target_rgb, th.rand_like(source_uncertainty.detach())*0.01, th.rand_like(target_uncertainty.detach())*0.01, delta_t
                source, target, source_uncertainty.detach(), target_uncertainty.detach(), delta_t, input_mode.view(-1, 1, 1, 1)
            )

            print(f'Saving[{i+1}/{len(dataloader)}|{i+1/len(dataloader)*100:.2f}%]')

            grayscale             = target_rgb[:,0:1] * 0.299 + target_rgb[:,1:2] * 0.587 + target_rgb[:,2:3] * 0.114
            target_rgb_highlited  = grayscale * (1 - target_uncertainty) 
            target_rgb_highlited += grayscale * target_uncertainty * 0.3333333 
            target_rgb_highlited  = target_rgb_highlited + to_rgb_object(target_uncertainty, 5) * 0.6666666

            width  = size[1] * 3 + 18*4
            height = size[0] * 2 + 18*3

            img = th.ones((3, height, width), device = device) * 0.2
            img[:,18:size[0]+18, 18:size[1]+18]                 = preprocess(source_rgb, size=size, add_text=add_text, text="GT RGB")[0]
            img[:,18:size[0]+18, 18*2+size[1]:18*2+size[1]*2]   = preprocess(source_depth, size = size, add_text=add_text, text="GT Depth")[0]
            img[:,18:size[0]+18, 18*3+size[1]*2:18*3+size[1]*3] = preprocess(target_rgb_highlited, size = size, add_text=add_text, text="Uncertainty masked")[0]

            img[:,-size[0]-18:-18, 18:size[1]+18]                 = preprocess(output_rgb, size=size, add_text=add_text, text="RGB Output")[0]
            img[:,-size[0]-18:-18, 18*2+size[1]:18*2+size[1]*2]   = preprocess(output_depth, size=size, add_text=add_text, text="Depth Output")[0]
            img[:,-size[0]-18:-18, 18*3+size[1]*2:18*3+size[1]*3] = preprocess(source_uncertainty, size = size, add_text=add_text, text="Uncertainty Output")[0]

            img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
            cv2.imwrite(f'background-grid-{i:04d}.jpg', img)

            if individual_views:
                cv2.imwrite(f'background-input-rgb-{i:04d}-rgb.jpg', rearrange(preprocess(source_rgb, size=size, add_text=add_text, text="GT RGB")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'background-input-depth-{i:04d}-{t+3:03d}-depth.jpg', rearrange(preprocess(source_depth, size=size, add_text=add_text, text="GT Depth")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'background-input-rgb-masked-{i:04d}-{t+3:03d}-rgb.jpg', rearrange(preprocess(target_rgb_highlited, size=size, add_text=add_text, text="Uncertainty masked")[0] * 255, 'c h w -> h w c').cpu().numpy())

                cv2.imwrite(f'background-output-rgb-{i:04d}-{t+3:03d}-rgb.jpg', rearrange(preprocess(output_rgb, size=size, add_text=add_text, text="RGB Output")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'background-output-depth-{i:04d}-{t+3:03d}-depth.jpg', rearrange(preprocess(output_depth, size=size, add_text=add_text, text="Depth Output")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'background-output-uncertainty-{i:04d}-{t+3:03d}-uncertainty.jpg', rearrange(preprocess(source_uncertainty, size=size, add_text=add_text, text="Uncertainty Output")[0] * 255, 'c h w -> h w c').cpu().numpy())





def save_objects(cfg: Configuration, dataset: Dataset, file, size, add_text, individual_views, mask = False, export_latent = False, input_mask = False):

    np.random.seed(1234)
    th.manual_seed(1234)

    #assert(cfg.sequence_len == 2)
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociPretrainerDataModule(cfg)
    dataloader  = data_module.val_dataloader()

    if file != '':
        model = LociPretrainerModule.load_from_checkpoint(file, cfg=cfg, strict=False).to(device)
    else:
        model = LociPretrainerModule(cfg).to(device)

    # create model 
    net = model.net
    net.eval()

    mask_center = MaskCenter(cfg_net.input_size).to(device)
    gaus2d      = Gaus2D(cfg_net.input_size).to(device)

    with th.no_grad():
        for i, input in enumerate(dataloader):

            input_rgb   = input[0].to(device)
            input_depth = input[1].to(device)
            input_instance_mask = input[2].to(device)

            results = net(input_rgb, input_depth, input_instance_mask, iterations=cfg.pretrainer_iterations, mode=cfg.pretraining_mode)

            if export_latent:
                gestalt  = results['gestalt'].cpu().numpy()
                position = results['position'].cpu().numpy()

                # save using pickle
                with open(f'out/{cfg.model_path}/latent-states-{i:04d}.pkl', 'wb') as f:
                    pickle.dump({'gestalt': gestalt, 'position': position}, f)

            print(f'Saving[{(i+1)*100/len(dataloader):.2f}%/{i+1}/{len(dataloader)}]')

            xy_std = th.cat(mask_center(input_instance_mask), dim=1)
            pos2d  = gaus2d(xy_std)

            grayscale            = input_rgb[:,0:1] * 0.299 + input_rgb[:,1:2] * 0.587 + input_rgb[:,2:3] * 0.114
            highlited_input_rgb  = grayscale * (1 - input_instance_mask) 
            highlited_input_rgb += grayscale * input_instance_mask * 0.3333333 
            highlited_input_rgb  = highlited_input_rgb + to_rgb_object(input_instance_mask, 5) * 0.333333 + to_rgb_object(pos2d, 1) * 0.333333

            norm_depth = th.sigmoid(results['depth'])

            width  = size[1] * 3 + 18*4
            height = size[0] * 2 + 18*3

            img = th.ones((3, height, width), device = device) * 0.2
            img[:,18:size[0]+18, 18:size[1]+18]                 = preprocess(input_rgb, size=size, add_text=add_text, text="GT RGB")[0]
            img[:,18:size[0]+18, 18*2+size[1]:18*2+size[1]*2]   = preprocess(input_depth, size = size, add_text=add_text, text="GT Depth")[0]
            img[:,18:size[0]+18, 18*3+size[1]*2:18*3+size[1]*3] = preprocess(highlited_input_rgb, size = size, add_text=add_text, text="GT Masked Input")[0]

            if mask:
                img[:,size[0]+36:size[0]*2+36, 18:size[1]+18]                 = preprocess(results['rgb'] * results['mask'], size=size, add_text=add_text, text="RGB Output")[0]
                img[:,size[0]+36:size[0]*2+36, 18*2+size[1]:18*2+size[1]*2]   = preprocess(norm_depth * results['mask'], size=size, add_text=add_text, text="Depth Output")[0]
            elif input_mask:
                img[:,size[0]+36:size[0]*2+36, 18:size[1]+18]                 = preprocess(results['rgb'] * input_instance_mask, size=size, add_text=add_text, text="RGB Output")[0]
                img[:,size[0]+36:size[0]*2+36, 18*2+size[1]:18*2+size[1]*2]   = preprocess(norm_depth * input_instance_mask, size=size, add_text=add_text, text="Depth Output")[0]
            else:
                img[:,size[0]+36:size[0]*2+36, 18:size[1]+18]                 = preprocess(results['rgb'], size=size, add_text=add_text, text="RGB Output")[0]
                img[:,size[0]+36:size[0]*2+36, 18*2+size[1]:18*2+size[1]*2]   = preprocess(norm_depth, size=size, add_text=add_text, text="Depth Output")[0]

            img[:,size[0]+36:size[0]*2+36, 18*3+size[1]*2:18*3+size[1]*3] = preprocess(results['mask'], size = size, add_text=add_text, text="Mask Output")[0]

            img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
            cv2.imwrite(f'object-grid-{i:04d}.jpg', img)

            if individual_views:
                cv2.imwrite(f'object-input-rgb-{i:04d}-rgb.jpg', rearrange(preprocess(input_rgb, size=size, add_text=add_text, text="RGB Input")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'object-input-depth-{i:04d}-depth.jpg', rearrange(preprocess(input_depth, size=size, add_text=add_text, text="Depth Input")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'object-input-masked-{i:04d}-masked.jpg', rearrange(preprocess(highlited_input_rgb, size=size, add_text=add_text, text="GT Masked Input")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'object-output-rgb-{i:04d}-rgb.jpg', rearrange(preprocess(results['rgb'] * results['mask'], size=size, add_text=add_text, text="RGB Output")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'object-output-depth-{i:04d}-depth.jpg', rearrange(preprocess(results['depth'] * results['mask'], size=size, add_text=add_text, text="Depth Output")[0] * 255, 'c h w -> h w c').cpu().numpy())
                cv2.imwrite(f'object-output-mask-{i:04d}-mask.jpg', rearrange(preprocess(results['mask'], size=size, add_text=add_text, text="Mask Output")[0] * 255, 'c h w -> h w c').cpu().numpy())


def save_masks(cfg: Configuration, dataset: Dataset, file, size, add_text, individual_views, mask = False, export_latent = False):

    np.random.seed(1234)
    th.manual_seed(1234)

    #assert(cfg.sequence_len == 2)
    cfg.pretraining_mode = "mask"
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociPretrainerDataModule(cfg)
    dataloader  = data_module.val_dataloader()

    if file != '':
        model = LociPretrainerModule.load_from_checkpoint(file, cfg=cfg, strict=False).to(device)
    else:
        model = LociPretrainerModule(cfg).to(device)

    # create model 
    model.net.eval()

    mask_center = MaskCenter(cfg_net.input_size).to(device)
    gaus2d      = Gaus2D(cfg_net.input_size).to(device)

    with th.no_grad():
        for i, input in enumerate(dataloader):

            input_rgb   = input[0].to(device)
            input_depth = input[1].to(device)
            input_instance_mask = input[2].to(device)

            results = model(input_rgb, input_depth, input_instance_mask)

            if export_latent:
                gestalt  = results['gestalt'].cpu().numpy()
                position = results['position'].cpu().numpy()

                # save using pickle
                with open(f'out/{cfg.model_path}/latent-states-{i:04d}.pkl', 'wb') as f:
                    pickle.dump({'gestalt': gestalt, 'position': position}, f)

            print(f'Saving[{(i+1)*100/len(dataloader):.2f}%/{i+1}/{len(dataloader)}]')

            xy_std = model.net.mask_pretrainer.mask_center(input_instance_mask)
            pos2d  = gaus2d(xy_std)
            
            width  = size[1] * 2 + 18 * 3
            height = size[0] * 2 + 18 * 3

            img = th.ones((3, height, width), device=device) * 0.2
            img[:, 18:size[0]+18, 18:size[1]+18] = preprocess(input_instance_mask, size=size, add_text=True, text="GT Mask")[0]
            img[:, 18*2+size[0]:18*2+size[0]*2, 18:size[1]+18] = preprocess(results['mask'][:,0:1], size=size, add_text=True, text="Mask Output")[0]

            img[:, 18:size[0]+18, -size[1]-18:-18] = preprocess(th.abs(results['mask'][:,0:1] - input_instance_mask), normalize=True, size=size, add_text=True, text="Error")[0]
            img[:, 18*2+size[0]:18*2+size[0]*2, -size[1]-18:-18] = preprocess(pos2d, size=size, add_text=True, text="Position")[0]

            img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
            cv2.imwrite(f'mask-grid-{i:04d}.jpg', img)


def save_depth(cfg: Configuration, dataset: Dataset, file, size, add_text, individual_views, mask = False, export_latent = False):

    np.random.seed(1234)
    th.manual_seed(1234)

    #assert(cfg.sequence_len == 2)
    cfg.pretraining_mode = "depth"
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociPretrainerDataModule(cfg)
    dataloader  = data_module.val_dataloader()

    if file != '':
        model = LociPretrainerModule.load_from_checkpoint(file, cfg=cfg, strict=False).to(device)
    else:
        model = LociPretrainerModule(cfg).to(device)

    # create model 
    model.net.eval()

    mask_center = MaskCenter(cfg_net.input_size).to(device)
    gaus2d      = Gaus2D(cfg_net.input_size).to(device)

    with th.no_grad():
        for i, input in enumerate(dataloader):

            input_rgb   = input[0].to(device)
            input_depth = input[1].to(device)
            input_instance_mask = input[2].to(device)

            results = model(input_rgb, input_depth, input_instance_mask)

            input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
            input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
            input_depth_std  = th.sqrt(
                th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
                (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
            )

            input_depth = th.sigmoid(((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask) * input_instance_mask

            if export_latent:
                gestalt  = results['gestalt'].cpu().numpy()
                position = results['position'].cpu().numpy()

                # save using pickle
                with open(f'out/{cfg.model_path}/latent-states-{i:04d}.pkl', 'wb') as f:
                    pickle.dump({'gestalt': gestalt, 'position': position}, f)

            print(f'Saving[{(i+1)*100/len(dataloader):.2f}%/{i+1}/{len(dataloader)}]')

            xy_std = model.net.mask_pretrainer.mask_center(input_instance_mask)
            pos2d  = gaus2d(xy_std)
            
            width  = size[1] * 2 + 18 * 3
            height = size[0] * 2 + 18 * 3

            img = th.ones((3, height, width), device=device) * 0.2
            img[:, 18:size[0]+18, 18:size[1]+18] = preprocess(input_depth, size=size, add_text=True, text="Input Depth")[0]
            img[:, 18*2+size[0]:18*2+size[0]*2, 18:size[1]+18] = preprocess(th.sigmoid(results['depth']) * input_instance_mask, size=size, add_text=True, text="Depth Output")[0]

            img[:, 18:size[0]+18, -size[1]-18:-18] = preprocess(th.abs(th.sigmoid(results['depth']) * input_instance_mask - input_depth), normalize=True, size=size, add_text=True, text="Error")[0]
            img[:, 18*2+size[0]:18*2+size[0]*2, -size[1]-18:-18] = preprocess(pos2d, size=size, add_text=True, text="Position")[0]

            img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
            cv2.imwrite(f'depth-grid-{i:04d}.jpg', img)

def save_rgb(cfg: Configuration, dataset: Dataset, file, size, add_text, individual_views, mask = False, export_latent = False):

    np.random.seed(1234)
    th.manual_seed(1234)

    #assert(cfg.sequence_len == 2)
    cfg.pretraining_mode = "rgb"
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociPretrainerDataModule(cfg)
    dataloader  = data_module.val_dataloader()

    if file != '':
        model = LociPretrainerModule.load_from_checkpoint(file, cfg=cfg, strict=False).to(device)
    else:
        model = LociPretrainerModule(cfg).to(device)

    # create model 
    model.net.eval()

    mask_center = MaskCenter(cfg_net.input_size).to(device)
    gaus2d      = Gaus2D(cfg_net.input_size).to(device)

    with th.no_grad():
        for i, input in enumerate(dataloader):

            input_rgb   = input[0].to(device)
            input_depth = input[1].to(device)
            input_instance_mask = input[2].to(device)

            results = model(input_rgb, input_depth, input_instance_mask)

            input_depth_mean = th.sum(input_depth * input_instance_mask, dim=(1,2,3), keepdim=True) 
            input_depth_mean = input_depth_mean / (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
            input_depth_std  = th.sqrt(
                th.sum((input_depth - input_depth_mean)**2 * input_instance_mask, dim=(1,2,3), keepdim=True) / 
                (th.sum(input_instance_mask, dim=(1,2,3), keepdim=True) + 1e-6)
            )

            input_depth = th.sigmoid(((input_depth - input_depth_mean) / (input_depth_std + 1e-6)) * input_instance_mask) * input_instance_mask

            if export_latent:
                gestalt  = results['gestalt'].cpu().numpy()
                position = results['position'].cpu().numpy()

                # save using pickle
                with open(f'out/{cfg.model_path}/latent-states-{i:04d}.pkl', 'wb') as f:
                    pickle.dump({'gestalt': gestalt, 'position': position}, f)

            print(f'Saving[{(i+1)*100/len(dataloader):.2f}%/{i+1}/{len(dataloader)}]')

            xy_std = model.net.mask_pretrainer.mask_center(input_instance_mask)
            pos2d  = gaus2d(xy_std)
            
            width  = size[1] * 2 + 18 * 3
            height = size[0] * 2 + 18 * 3

            img = th.ones((3, height, width), device=device) * 0.2
            img[:, 18:size[0]+18, 18:size[1]+18] = preprocess(input_rgb * input_instance_mask, size=size, add_text=True, text="Input RGB")[0]
            img[:, 18*2+size[0]:18*2+size[0]*2, 18:size[1]+18] = preprocess(input_depth, size=size, add_text=True, text="GT Depth")[0]

            img[:, 18:size[0]+18, -size[1]-18:-18] = preprocess(th.abs(results['rgb'] - input_rgb) * input_instance_mask, normalize=True, size=size, add_text=True, text="Error")[0]
            img[:, 18*2+size[0]:18*2+size[0]*2, -size[1]-18:-18] = preprocess(results['rgb'] * input_instance_mask, size=size, add_text=True, text="RGB Output")[0]

            img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
            cv2.imwrite(f'rgb-grid-{i:04d}.jpg', img)

def save_proposal(cfg: Configuration, file, size, add_text, individual_views, mask = False, export_latent = False):

    np.random.seed(1234)
    th.manual_seed(1234)

    #assert(cfg.sequence_len == 2)
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    data_module = LociProposalDataModule(cfg)
    dataloader  = data_module.val_dataloader()

    if file != '':
        model = LociProposalModule.load_from_checkpoint(file, cfg=cfg, strict=False).to(device)
    else:
        model = LociProposalModule(cfg).to(device)

    # create model 
    model.net.eval()

    gaus2d = Gaus2D(cfg_net.input_size).to(device)

    with th.no_grad():
        for i, input in enumerate(dataloader):

            input_rgb   = input[0].to(device)
            input_depth = input[1].to(device)
            input_instance_mask = input[2].to(device)

            results = model(input_rgb, input_depth, input_instance_mask)

            print(f'Saving[{(i+1)*100/len(dataloader):.2f}%/{i+1}/{len(dataloader)}]: {results["iou"]:.2f}%')

            input_instance_mask = reduce(input_instance_mask, 'b c h w -> b 1 h w', 'max')
            input_instance_mask = repeat(input_instance_mask, 'b 1 h w -> b c h w', c = 3)

            color_mask = th.zeros_like(input_rgb)
            for o in range(results['mask'].shape[1]):
                color_mask += to_rgb_object(results['mask'][0:1,o], o)

            fg_mask = reduce(results['mask'], 'b o h w -> b 1 h w', 'max')

            grayscale            = input_rgb[:,0:1] * 0.299 + input_rgb[:,1:2] * 0.587 + input_rgb[:,2:3] * 0.114
            highlited_input_rgb  = grayscale * (1 - fg_mask) + (color_mask * 0.6666 + grayscale * 0.3333) * fg_mask 

            positions = th.zeros_like(input_rgb)
            all_pos2d = th.zeros_like(input_rgb)
            for n in range(cfg_net.num_slots):
                pos2d = gaus2d(results['position'][0:1,n])
                all_pos2d = th.maximum(pos2d, all_pos2d)
                positions = positions * (1 - pos2d) + pos2d * to_rgb_object(pos2d, n)

            positions = grayscale * (1 - all_pos2d) + all_pos2d * (positions * 0.666 + grayscale * 0.333)
            
            width  = size[1] * 3 + 18*4
            height = size[0] * 2 + 18*3

            img = th.ones((3, height, width), device = device) * 0.2
            img[:,18:size[0]+18, 18:size[1]+18]                 = preprocess(input_rgb, size=size, add_text=add_text, text="RGB Input")[0]
            img[:,18:size[0]+18, 18*2+size[1]:18*2+size[1]*2]   = preprocess(input_depth, size = size, add_text=add_text, text="GT Depth")[0]
            img[:,18:size[0]+18, 18*3+size[1]*2:18*3+size[1]*3] = preprocess(highlited_input_rgb, size = size, add_text=add_text, text="RGB Masked")[0]

            img[:,size[0]+36:size[0]*2+36, 18:size[1]+18]                 = preprocess(results['softmask'], size=size, add_text=add_text, text="Reg Mask")[0]
            img[:,size[0]+36:size[0]*2+36, 18*2+size[1]:18*2+size[1]*2]   = preprocess(color_mask, size=size, add_text=add_text, text="Mask")[0]
            img[:,size[0]+36:size[0]*2+36, 18*3+size[1]*2:18*3+size[1]*3] = preprocess(positions, size=size, add_text=add_text, text="Positions")[0]

            img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
            cv2.imwrite(f'proposal-grid-{i:04d}.jpg', img)

