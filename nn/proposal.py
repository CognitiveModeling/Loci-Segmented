import torch as th
import numpy as np
import torchvision as tv
from torch import nn
from utils.utils import BatchToSharedObjects, SharedObjectsToBatch, LambdaModule, MultiArgSequential, Binarize, Prioritize
from einops import rearrange, repeat, reduce
from utils.optimizers import Ranger
from nn.convnext import MemoryEfficientBottleneck, ConvNeXtBlock, ConvNeXtUnet, ConvNeXtPatchEmbeddingEncoder
import torch.nn.functional as F
from utils.loss import MaskedL1SSIMLoss
from nn.downscale import MemoryEfficientPatchDownScale
from nn.eprop_gate_l0rd import ReTanh
from nn.mask_decoder import MaskPretrainer, PositionPooling, Gaus2D, MaskCenter
from nn.convnext import SkipConnection
from scipy.optimize import linear_sum_assignment
from nn.upscale import MemoryEfficientUpscaling
from einops.layers.torch import Rearrange, Reduce
from skimage import measure
import cv2

class Grid2D(nn.Module):
    def __init__(self, size=None):
        super(Grid2D, self).__init__()
        self.size = size
        self.register_buffer("grid_x", th.zeros(1,1,1,1), persistent=False)
        self.register_buffer("grid_y", th.zeros(1,1,1,1), persistent=False)

        if size is not None:
            self.update_grid(size)

    def update_grid(self, size):
        if size != self.grid_x.shape[2:]:
            self.size = size
            H, W = size

            self.grid_x = th.arange(W, device=self.grid_x.device)
            self.grid_y = th.arange(H, device=self.grid_x.device)

            self.grid_x = (self.grid_x / (W-1)) * 2 - 1
            self.grid_y = (self.grid_y / (H-1)) * 2 - 1

            self.grid_x = self.grid_x.view(1, 1, 1, -1).expand(1, 1, H, W).clone()
            self.grid_y = self.grid_y.view(1, 1, -1, 1).expand(1, 1, H, W).clone()

class InstanceSegmentationLoss(th.nn.Module):
    def __init__(self):
        super(InstanceSegmentationLoss, self).__init__()

    @staticmethod
    def compute_iou_matrix(gt_masks, pred_masks):
        N = gt_masks.shape[1]
        M = pred_masks.shape[1]

        # unsqueeze to broadcast
        # gt_masks: (B, N, H, W) -> (B, 1, N, H, W)
        # pred_masks: (B, M, H, W) -> (B, M, 1, H, W)
        gt_masks = gt_masks.unsqueeze(1)
        pred_masks = pred_masks.unsqueeze(2)
        
        # Calculate the intersection (N, M)
        intersection = (gt_masks * pred_masks).sum(dim=(3, 4))
        
        # Calculate the union (N, M)
        union = gt_masks.sum(dim=(3, 4)) + pred_masks.sum(dim=(3, 4)) - intersection
        
        # Calculate IoU
        iou_matrix = intersection / (union + 1e-8)
        return iou_matrix

    @staticmethod
    def hungarian_assignment(cost_matrix):
        row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        return row_indices, col_indices

    def forward(self, gt_masks, bg_mask, pred_logits):
        pred_logits = th.cat((th.zeros_like(pred_logits[:,:1]), pred_logits), dim=1)
        pred_masks  = th.softmax(pred_logits, dim=1).detach()

        # Compute the IoU matrix
        iou_matrix = self.compute_iou_matrix(gt_masks, pred_masks)
        
        # Convert IoU to 'cost'
        cost_matrix = 1 - iou_matrix

        selected_logits = []
        
        B, _, H, W = gt_masks.shape
        mean_iou = 0

        loss = []

        valid_mask = reduce(gt_masks, 'b c h w -> b c', 'max') 

        weights = 1 / (th.sum(gt_masks, dim=(2, 3)) + 1)
        weights = weights * valid_mask + (1 - valid_mask) / (H * W)

        for i in range(cost_matrix.shape[0]):
            # Hungarian assignment
            pred_idx, gt_idx = self.hungarian_assignment(cost_matrix[i])
            
            # Update target_idx using the gt_idx
            target_idx = th.zeros((1, H, W), dtype=th.long, device=gt_masks.device)
            for cls, idx in enumerate(gt_idx):
                target_idx[0] = target_idx[0] + gt_masks[i, idx] * cls

            mean_iou += iou_matrix[i, pred_idx, gt_idx].sum() / (valid_mask[i].sum() + 1e-16)
            
            loss.append(F.cross_entropy(pred_logits[i][pred_idx].unsqueeze(0), target_idx.detach(), weight=weights[i][gt_idx].detach()))

        return th.mean(th.stack(loss)), mean_iou / B

class PositionProposal(nn.Module):
    def __init__(
        self, 
        num_slots, 
        encoder_blocks,
        decoder_blocks,
        base_channels,
        depth_input=False,
    ):
        super(PositionProposal, self).__init__()

        in_channels   = 3 if depth_input else 5
        out_channels  = num_slots * 2
        
        self.encoder = ConvNeXtPatchEmbeddingEncoder(in_channels, base_channels, encoder_blocks, return_features=True)

        self.layer0 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 8) for _ in range(decoder_blocks[3])],
            MemoryEfficientUpscaling(base_channels * 8, base_channels * 4, scale_factor=2) if decoder_blocks[3] > 0 else nn.Identity(),
        )

        self.merge1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 4) for _ in range(decoder_blocks[1])],
            MemoryEfficientUpscaling(base_channels * 4, base_channels * 2, scale_factor=2),
        )

        self.merge2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            *[ConvNeXtBlock(base_channels * 2) for _ in range(decoder_blocks[1])],
            MemoryEfficientUpscaling(base_channels * 2, base_channels, scale_factor=2),
        )

        self.merge3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.layer3 = nn.Sequential(
            *[ConvNeXtBlock(base_channels) for _ in range(decoder_blocks[1])],
            MemoryEfficientUpscaling(base_channels, out_channels, scale_factor=4),
        )

        self.to_possition = nn.Conv2d(out_channels, 5, kernel_size=1)

        self.hyper_network = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(base_channels * 8, base_channels * 8),
            nn.SiLU(),
            nn.Linear(base_channels * 8, base_channels * 8),
            nn.SiLU(),
            nn.Linear(base_channels * 8, out_channels*num_slots),
            Rearrange('b (n c) -> b n c', n=num_slots),
        )

    def forward(self, input: th.Tensor):

        features = self.encoder(input)

        x = self.layer0(features[0])
        x = self.layer1(self.merge1(th.cat((x, features[1]), dim=1)))
        x = self.layer2(self.merge2(th.cat((x, features[2]), dim=1)))
        x = self.layer3(self.merge3(th.cat((x, features[3]), dim=1)))

        position = self.to_possition(x)

        masks = self.hyper_network(features[0]).bmm(rearrange(x, 'b c h w -> b c (h w)'))
        masks = rearrange(masks, 'b c (h w) -> b c h w', h=x.shape[2], w=x.shape[3])
        
        return position, masks

def hardmax_with_bg(mask_logits):
    # Add a background class with zero logits
    B, N, H, W = mask_logits.shape
    mask_logits_with_bg = th.cat((th.zeros(B, 1, H, W, device=mask_logits.device, dtype=mask_logits.dtype), mask_logits), dim=1)
    
    # Find the argmax along the class dimension
    max_indices = th.argmax(mask_logits_with_bg, dim=1)
    
    # Create an output tensor filled with zeros
    max_equiv = th.zeros_like(mask_logits_with_bg)
    
    # Efficiently set ones at the maximum indices
    max_equiv.scatter_(1, max_indices.unsqueeze(1), 1)
    
    # Remove the added background class
    max_equiv = max_equiv[:, 1:, :, :]
    return max_equiv

def largest_connected_component(masks):
    B, N, H, W = masks.shape
    largest_components = np.zeros((B, N, H, W), dtype=np.float32)
    
    for b in range(B):
        for n in range(N):
            mask = masks[b, n].cpu().numpy()
            labels = measure.label(mask)
            if labels.max() == 0:  # No connected components found
                continue
            component_sizes = np.bincount(labels.ravel())
            component_sizes[0] = 0  # Ignore background
            largest_component = np.argmax(component_sizes)
            largest_components[b, n] = (labels == largest_component).astype(np.float32)
            
    return th.from_numpy(largest_components).to(masks.device)


def fill_gaps(masks, kernel_size=5):
    B, N, H, W = masks.shape
    filled_masks = np.zeros((B, N, H, W), dtype=np.float32)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    for b in range(B):
        for n in range(N):
            mask = masks[b, n].cpu().numpy()
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            filled_masks[b, n] = mask_closed
            
    return th.from_numpy(filled_masks).to(masks.device)

def remove_small_artifacts(masks, kernel_size=5):
    B, N, H, W = masks.shape
    
    # Create the kernel for erosion and dilation
    kernel = th.ones((1, 1, kernel_size, kernel_size), device=masks.device)
    
    # Apply erosion
    eroded_masks = F.conv2d(masks.view(B * N, 1, H, W), kernel, padding=kernel_size // 2) >= (kernel_size ** 2)
    eroded_masks = eroded_masks.view(B, N, H, W).float()

    cleaned_masks = masks * reduce(eroded_masks, 'b n h w -> b n 1 1', 'max')
    return cleaned_masks


class LociProposal(nn.Module):
    def __init__(
        self,
        size,
        num_slots, 
        encoder_blocks,
        decoder_blocks,
        base_channels,
        depth_input=False,
    ):
        super(LociProposal, self).__init__()
        self.num_slots = num_slots
        self.depth_input = depth_input
        
        latent_size = [size[1] // 16, size[0] // 16]

        self.mask_center = MaskCenter(size)
        self.gaus2d      = Gaus2D(latent_size)

        self.position_estimator = PositionProposal(num_slots, encoder_blocks, decoder_blocks, base_channels, depth_input)
        self.grid = Grid2D()
        
        self.mask_loss = InstanceSegmentationLoss()

    def forward(self, gt_masks, gt_depth, gt_rgb = None, fg_mask = None):
        self.grid.update_grid(gt_depth.shape[2:])

        B, _, H, W = gt_depth.shape
        N = gt_masks.shape[1]
        d = gt_masks.device
        assert gt_masks.shape == (B, self.num_slots, H, W)
        
        with th.no_grad():
            gt_masks = (gt_masks > 0.8).float()
            valid_mask = reduce(gt_masks, 'b n h w -> (b n) 1', 'max', n=N)

            depth_mask = reduce((gt_depth > 0).float(), 'b 1 h w -> b 1 1 1', 'max')
            depth_mask = th.cat((
                th.ones_like(depth_mask), 
                th.ones_like(depth_mask), 
                depth_mask,
                th.ones_like(depth_mask), 
                depth_mask,
            ), dim=1)

            gt_z = th.sum(gt_depth * gt_masks, dim=(2,3), keepdim=True) / (th.sum(gt_masks, dim=(2,3), keepdim=True) + 1e-8)
            gt_std_z = th.sqrt(th.sum((gt_depth - gt_z)**2 * gt_masks, dim=(2,3)) / (th.sum(gt_masks, dim=(2,3)) + 1e-8))

            gt_z = rearrange(gt_z, 'b n 1 1 -> (b n) 1')
            gt_std_z = rearrange(gt_std_z, 'b n -> (b n) 1')

            if fg_mask is None:
                fg_mask  = reduce(gt_masks, 'b n h w -> b 1 h w', 'max', n=N) 

            gt_masks = rearrange(gt_masks, 'b n h w -> (b n) 1 h w', n=N)
            gt_xy, gt_std = self.mask_center(gt_masks)

            gt_positions = th.cat((gt_xy, gt_z, gt_std, gt_std_z), dim=1) * valid_mask
            gt_positions = rearrange(gt_positions, 'b c -> b c 1 1')

            target = reduce(gt_masks * gt_positions, '(b n) c h w -> b c h w', 'max', n=N) # TODO (make sure mask do not overlap)

            if self.depth_input:
                input  = th.cat((gt_depth * fg_mask, self.grid.grid_x * fg_mask, self.grid.grid_y * fg_mask), dim=1) 
            else:
                input  = th.cat((gt_rgb * fg_mask, self.grid.grid_x * fg_mask, self.grid.grid_y * fg_mask), dim=1)

            gt_masks = rearrange(gt_masks, '(b n) 1 h w -> b n h w', n=N)

        output, mask_logits = self.position_estimator(input)

        mask_loss, mean_iou = self.mask_loss(gt_masks, 1 - fg_mask, mask_logits)

        if self.depth_input:
            loss = th.mean((output - target)**2)
        else:
            loss = th.mean((output - target)**2 * depth_mask)

        if self.training:
            return {
                'regularizer_loss': loss,
                'iou': mean_iou.detach() * 100,
                'mask_loss': mask_loss*0.01,
                'position': th.zeros_like(gt_positions),
                'mask': th.zeros_like(gt_masks), 
                'softmask': th.clip(output[:,:3], 0, 1),
            }   


        masks = hardmax_with_bg(mask_logits)
        masks = largest_connected_component(masks)
        masks = fill_gaps(masks)
        masks = remove_small_artifacts(masks)

        with th.no_grad():
            valid_mask = reduce(masks, 'b n h w -> (b n) 1', 'max', n=N)

            z = th.sum(gt_depth * masks, dim=(2,3), keepdim=True) / (th.sum(masks, dim=(2,3), keepdim=True) + 1e-8)
            z = rearrange(z, 'b n 1 1 -> (b n) 1')

            xy, std = self.mask_center(rearrange(masks, 'b n h w -> (b n) 1 h w', n=N))

            positions = th.cat((xy, z, std), dim=1) * valid_mask
            positions = rearrange(positions, '(b n) c -> b n c', n=N)

        return {
            'regularizer_loss': loss,
            'iou': mean_iou.detach() * 100,
            'mask_loss': mask_loss*0.01,
            'position': positions,
            'mask': masks, 
            'softmask': th.clip(output[:,:3], 0, 1),
        }   

        
