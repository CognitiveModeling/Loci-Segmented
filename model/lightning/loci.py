import pytorch_lightning as pl
import torch as th
import numpy as np
from torch.utils.data import DataLoader
from utils.io import UEMA, Timer
from utils.optimizers import Ranger
from model.loci import Loci
from utils.loss import MaskedL1SSIMLoss, MaskedYCbCrL2SSIMLoss
from einops import rearrange, repeat, reduce
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

# detach all values recursively in the given dict
def detach_dict(d):
    for key, value in d.items():
        if isinstance(value, th.Tensor):
            d[key] = value.detach()
        elif isinstance(value, dict):
            detach_dict(value)

class LociModule(pl.LightningModule):
    def __init__(self, cfg, pretraining_state={}):
        super().__init__()
        self.cfg = cfg
        self.own_loggers = {}
        self.timer = Timer()

        cfg.model.teacher_forcing = cfg.teacher_forcing

        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        self.val_metrics = {}

        self.net = Loci(cfg.model)

        self.l1ssim  = MaskedL1SSIMLoss()
        self.rgbloss = MaskedL1SSIMLoss() if cfg.model.rgb_loss == 'l1ssim' else MaskedYCbCrL2SSIMLoss()

        self.last_input = None
        self.last_rgb = None
        self.last_depth = None
        self.output_last = None

        self.load_component_state("encoder", pretraining_state)
        self.load_component_state("decoder", pretraining_state)
        self.load_component_state("background", pretraining_state)

        self.net.object_discovery.load_pretrained(pretraining_state)

        self.teacher_forcing = ((cfg.teacher_forcing // cfg.backprop_steps) * cfg.backprop_steps + 1) if cfg.sequence_len > 1 else cfg.teacher_forcing

        for param in self.net.background.parameters():
            param.requires_grad_(False)

        self.num_updates = -1

        self.gt_mask_seq = None
        self.pred_mask_seq = None
        self.rec_mask_seq = None


    # conservative loading of pretraining states if they exist
    def load_component_state(self, component, pretraining_state):
        if component in pretraining_state:
            component_state = getattr(self.net, component).state_dict()
            for key, value in pretraining_state[component].items():
                if key in component_state:
                    component_state[key] = value
                else:
                    print("Not loading: ", key)

            for key, value in component_state.items():
                if key not in pretraining_state[component]:
                    print("New key: ", key)

            getattr(self.net, component).load_state_dict(component_state)

    def forward(self, **args):
        return self.net(**args)

    def log(self, name, value, on_step=False, on_epoch=True, prog_bar=False, logger=True):
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

    def compute_binarization(self, code, mask):

        slot_used = (reduce(mask, 'b (o c) h w -> (b o) c', 'max', o = self.cfg.model.num_slots) > 0.8).float()
        num_slots = th.mean(reduce(slot_used, '(b o) 1 -> b', 'sum', o = self.cfg.model.num_slots))
        code      = rearrange(code, 'b (o c) -> (b o) c', o = self.cfg.model.num_slots)
        slot_used = slot_used * th.ones_like(code)
        
        code_mean             = th.mean(th.clip(code, 0, 1))
        binarization_mean     = th.sum(th.minimum(th.abs(code), th.abs(1 - code))    * slot_used) / (th.sum(slot_used) + 1e-8)
        binarization_squared  = th.sum(th.minimum(th.abs(code), th.abs(1 - code))**2 * slot_used) / (th.sum(slot_used) + 1e-8)
        binarization_std      = th.sqrt(th.clip(binarization_squared - binarization_mean**2, min=0))
        
        return binarization_mean, binarization_std, code_mean, num_slots

    def model_step(self, batch, batch_idx, prefix):
        input_rgb, input_depth, time_steps, use_depth = batch
        output = self.output_last

        rgb_mask = th.ones_like(input_rgb[:, 0:1, 0:1])
        depth_mask = th.ones_like(input_depth[:, 0:1]) * use_depth.view(-1, 1, 1, 1, 1).float()

        if time_steps[0,0].item() == -self.teacher_forcing:
            self.net.reset_state()

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

            if self.cfg.model.inference_mode == "segmentation":
                bg_input        = th.cat((input_rgb[:,0], input_depth[:,0]), dim=1) if self.cfg.model.input_depth else input_rgb[:,0]
                uncertainty_cur = self.net.background.uncertainty_estimation(bg_input)[0]
                fg_mask         = (uncertainty_cur > 0.8).float()
                
                B, _, _, H, W = input_rgb.shape
                gt_masks = th.zeros((B, self.cfg.model.position_proposal.num_slots, H, W), device = input_rgb.device)
            
                results = self.net.proposal(gt_masks, input_depth[:,0], input_rgb[:,0], fg_mask = fg_mask) 
            
                seg_position = results['position']
                seg_mask     = results['mask']
                
                # sort by mask size
                seg_mask_sum = reduce(seg_mask, 'b o h w -> b o', 'sum')
                sorted_values, sorted_indices = th.sort(seg_mask_sum, dim=1, descending=True)
            
                # Using advanced indexing to sort the masks and positions
                sorted_seg_mask     = seg_mask[th.arange(seg_mask.size(0)).unsqueeze(1), sorted_indices]
                sorted_seg_position = seg_position[th.arange(seg_position.size(0)).unsqueeze(1), sorted_indices]
            
                sorted_seg_position = sorted_seg_position[:,:self.cfg.model.num_slots]
                sorted_seg_mask     = sorted_seg_mask[:,:self.cfg.model.num_slots]
            
                output['reconstruction']['position'] = rearrange(sorted_seg_position, 'b n c -> b (n c)')
                output['reconstruction']['mask']     = th.cat((sorted_seg_mask, 1 - reduce(results['mask'], 'b n h w -> b 1 h w', 'max')), dim=1)



        else:
            input_rgb   = th.cat((self.last_rgb, input_rgb), dim=1)
            input_depth = th.cat((self.last_depth, input_depth), dim=1)
            time_steps  = th.cat((self.last_time_step, time_steps), dim=1)
        
        # mask invalid / non existing depth values
        depth_mask = depth_mask * (input_depth > 0)

        self.net.detach()
        detach_dict(output)

        reconstruction_rgb_loss        = th.tensor(0.0, device=self.device)
        reconstruction_depth_loss      = th.tensor(0.0, device=self.device)
        reconstruction_rgb_l1_loss     = th.tensor(0.0, device=self.device)
        reconstruction_depth_l1_loss   = th.tensor(0.0, device=self.device)
        reconstruction_rgb_ssim_loss   = th.tensor(0.0, device=self.device)
        reconstruction_depth_sim_loss  = th.tensor(0.0, device=self.device)
        prediction_rgb_loss        = th.tensor(0.0, device=self.device)
        prediction_depth_loss      = th.tensor(0.0, device=self.device)
        prediction_rgb_l1_loss     = th.tensor(0.0, device=self.device)
        prediction_depth_l1_loss   = th.tensor(0.0, device=self.device)
        prediction_rgb_ssim_loss   = th.tensor(0.0, device=self.device)
        prediction_depth_sim_loss  = th.tensor(0.0, device=self.device)
        position_loss              = th.tensor(0.0, device=self.device)
        object_loss                = th.tensor(0.0, device=self.device)
        time_loss                  = th.tensor(0.0, device=self.device)

        detached_outputs = []
        for time_step in range(len(time_steps[0])-1):
            t = time_step if time_steps[0,time_step].item() >= 0 and self.cfg.sequence_len > 1 else 0
            output_last = output['prediction'] if time_steps[0,time_step].item() > 0 else output['reconstruction']
            output = self(
                input_rgb       = input_rgb[:, t],
                input_depth     = input_depth[:, t] if self.cfg.model.input_depth else output_last['output_depth'],
                bg_rgb_last     = output['prediction']['bg_rgb'],
                bg_depth_last   = output['prediction']['bg_depth'],
                rgb_last        = output_last['rgb'] if time_step > 0 else None, # will be recalculated 
                depth_raw_last  = output_last['depth_raw']  if time_step > 0 else None, # in the first time step
                mask_last       = output_last['mask'],
                mask_raw_last   = output_last['mask_raw'],
                occlusion_last  = output_last['occlusion'],
                position_last   = output_last['position'],
                gestalt_last    = output_last['gestalt'],
                priority_last   = output_last['priority'],
                teacher_forcing = time_steps[0, time_step] < 0,
                reset           = False,
                detach          = False,
                evaluate        = False, 
                test            = False,
            )
            detached_outputs.append(output)

            position_loss = position_loss + output['position_loss'] / len(time_steps[0])
            object_loss   = object_loss   + output['object_loss']   / len(time_steps[0])
            time_loss     = time_loss     + output['time_loss']     / len(time_steps[0])

            rgb_loss, rgb_l1, rgb_ssim      = self.rgbloss(output['reconstruction']['output_rgb'], input_rgb[:, t], rgb_mask)
            depth_loss, depth_l1, dept_ssim = self.l1ssim(output['reconstruction']['output_depth'], input_depth[:, t], depth_mask[:, t])

            reconstruction_rgb_loss   = reconstruction_rgb_loss   + rgb_loss   / len(time_steps[0])
            reconstruction_depth_loss = reconstruction_depth_loss + depth_loss / len(time_steps[0])

            reconstruction_rgb_l1_loss   = reconstruction_rgb_l1_loss   + rgb_l1   / len(time_steps[0])
            reconstruction_depth_l1_loss = reconstruction_depth_l1_loss + depth_l1 / len(time_steps[0])

            reconstruction_rgb_ssim_loss   = reconstruction_rgb_ssim_loss  + rgb_ssim  / len(time_steps[0])
            reconstruction_depth_sim_loss  = reconstruction_depth_sim_loss + dept_ssim / len(time_steps[0])

            if time_steps[0,time_step].item() >= 0:
                rgb_loss, rgb_l1, rgb_ssim      = self.rgbloss(output['prediction']['output_rgb'], input_rgb[:, t+1], rgb_mask)
                depth_loss, depth_l1, dept_ssim = self.l1ssim(output['prediction']['output_depth'], input_depth[:, t+1], depth_mask[:, t+1])

                prediction_rgb_loss   = prediction_rgb_loss   + rgb_loss   / len(time_steps[0])
                prediction_depth_loss = prediction_depth_loss + depth_loss / len(time_steps[0])

                prediction_rgb_l1_loss   = prediction_rgb_l1_loss   + rgb_l1   / len(time_steps[0])
                prediction_depth_l1_loss = prediction_depth_l1_loss + depth_l1 / len(time_steps[0])

                prediction_rgb_ssim_loss   = prediction_rgb_ssim_loss  + rgb_ssim  / len(time_steps[0])
                prediction_depth_sim_loss  = prediction_depth_sim_loss + dept_ssim / len(time_steps[0])

        loss = prediction_rgb_loss + prediction_depth_loss
        loss = loss + (reconstruction_rgb_loss + reconstruction_depth_loss) * self.cfg.model.encoder_regularizer
        loss = loss + self.cfg.model.position_regularizer * position_loss
        loss = loss + self.cfg.model.object_regularizer * object_loss
        loss = loss + self.cfg.model.time_regularizer * time_loss

        gestalt_binarization_mean, gestalt_binarization_std, gestalt_mean, num_slots = self.compute_binarization(output["reconstruction"]["gestalt"], output["reconstruction"]["mask"][:,:-1])

        self.log(f'{prefix}_loss',               loss.item())
        self.log(f'{prefix}_reconstruction_rgb_loss',          reconstruction_rgb_loss.item())
        self.log(f'{prefix}_reconstruction_depth_loss',        reconstruction_depth_loss.item())
        self.log(f'{prefix}_reconstruction_rgb_l1_loss',       reconstruction_rgb_l1_loss.item())
        self.log(f'{prefix}_reconstruction_depth_l1_loss',     reconstruction_depth_l1_loss.item())
        self.log(f'{prefix}_reconstruction_rgb_ssim_loss',     reconstruction_rgb_ssim_loss.item())
        self.log(f'{prefix}_reconstruction_depth_ssim_loss',   reconstruction_depth_sim_loss.item())
        self.log(f'{prefix}_reconstruction_gestalt_bin_mean',  gestalt_binarization_mean)
        self.log(f'{prefix}_reconstruction_gestalt_bin_std',   gestalt_binarization_std)
        self.log(f'{prefix}_reconstruction_gestalt_mean',      gestalt_mean)
        self.log(f'{prefix}_reconstruction_num_slots',         num_slots)

        if time_steps[0,-1].item() >= 0:

            binarization_mean, binarization_std, gestalt_mean, num_slots = self.compute_binarization(output["prediction"]["gestalt"], output["prediction"]["mask"][:,:-1])

            self.log(f'{prefix}_prediction_rgb_loss',               prediction_rgb_loss.item())
            self.log(f'{prefix}_prediction_depth_loss',             prediction_depth_loss.item())
            self.log(f'{prefix}_prediction_rgb_l1_loss',            prediction_rgb_l1_loss.item())
            self.log(f'{prefix}_prediction_depth_l1_loss',          prediction_depth_l1_loss.item())
            self.log(f'{prefix}_prediction_rgb_ssim_loss',          prediction_rgb_ssim_loss.item())
            self.log(f'{prefix}_prediction_depth_ssim_loss',        prediction_depth_sim_loss.item())
            self.log(f'{prefix}_prediction_binarization_mean',      binarization_mean)
            self.log(f'{prefix}_prediction_binarization_std',       binarization_std)
            self.log(f'{prefix}_prediction_gestalt_mean',           gestalt_mean)
            self.log(f'{prefix}_prediction_num_slots',              num_slots)
            self.log(f'{prefix}_prediction_position_gate',          th.mean(output['prediction']['position_gate']))
            self.log(f'{prefix}_prediction_gestalt_gate',           th.mean(output['prediction']['gestalt_gate']))
            self.log(f'{prefix}_prediction_position_gate_openings', th.mean((output['prediction']['position_gate'] > 0).float()))
            self.log(f'{prefix}_prediction_gestalt_gate_openings',  th.mean((output['prediction']['gestalt_gate'] > 0).float()))
            self.log(f'{prefix}_openings',                          self.net.get_openings())
            self.log(f'{prefix}_position_loss',                     position_loss.item())
            self.log(f'{prefix}_object_loss',                       object_loss.item())
            self.log(f'{prefix}_time_loss',                         time_loss.item())

        # detach everything in output
        for key in output:
            if isinstance(output[key], th.Tensor):
                output[key] = output[key].detach()

        self.last_rgb        = input_rgb[:,-1:].detach()
        self.last_depth      = input_depth[:,-1:].detach()
        self.last_time_step  = time_steps[:,-1:].detach()
        self.output_last     = output

        for i in range(len(detached_outputs)):
            detach_dict(detached_outputs[i])
        
        return loss, detached_outputs


    def training_step(self, batch, batch_idx):
        if self.cfg.sequence_len == 1:
            batch[0] = th.cat([batch[0] for _ in range(self.cfg.backprop_steps+1)], dim=1)
            batch[1] = th.cat([batch[1] for _ in range(self.cfg.backprop_steps+1)], dim=1)

        loss = self.model_step(batch, batch_idx, prefix='train')[0]
        time_steps = batch[2]

        if self.num_updates < self.trainer.global_step:
            self.num_updates = self.trainer.global_step
            itpersec = str(self.timer)

            print("Enc[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, depth: {:.2e}|{:.2e}|{:.2e}, obj: {:.1f}, binar: {:.2e}|{:.2e}|{:.3f}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                itpersec,
                float(self.own_loggers['train_loss']),
                float(self.own_loggers['train_reconstruction_rgb_loss']), 
                float(self.own_loggers['train_reconstruction_rgb_l1_loss']),
                float(self.own_loggers['train_reconstruction_rgb_ssim_loss']), 
                float(self.own_loggers['train_reconstruction_depth_loss']),
                float(self.own_loggers['train_reconstruction_depth_l1_loss']),
                float(self.own_loggers['train_reconstruction_depth_ssim_loss']),
                float(self.own_loggers['train_reconstruction_num_slots']),
                float(self.own_loggers['train_reconstruction_gestalt_bin_mean']),
                float(self.own_loggers['train_reconstruction_gestalt_bin_std']),
                float(self.own_loggers['train_reconstruction_gestalt_mean']),
            ), flush=True)

            if time_steps[0,-1].item() >= 0:

                print("Prd[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, depth: {:.2e}|{:.2e}|{:.2e}, obj: {:.1f}, binar: {:.2e}|{:.2e}|{:.3f}, Reg: {:.2e}|{:.2e}|{:.2e}, open: {:.2e}".format(
                    self.trainer.local_rank,
                    self.trainer.global_step,
                    self.trainer.current_epoch,
                    (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                    itpersec,
                    float(self.own_loggers['train_loss']),
                    float(self.own_loggers['train_prediction_rgb_loss']),
                    float(self.own_loggers['train_prediction_rgb_l1_loss']),
                    float(self.own_loggers['train_prediction_rgb_ssim_loss']),
                    float(self.own_loggers['train_prediction_depth_loss']),
                    float(self.own_loggers['train_prediction_depth_l1_loss']),
                    float(self.own_loggers['train_prediction_depth_ssim_loss']),
                    float(self.own_loggers['train_prediction_num_slots']),
                    float(self.own_loggers['train_prediction_binarization_mean']),
                    float(self.own_loggers['train_prediction_binarization_std']),
                    float(self.own_loggers['train_prediction_gestalt_mean']),
                    float(self.own_loggers['train_object_loss']),
                    float(self.own_loggers['train_time_loss']),
                    float(self.own_loggers['train_position_loss']),
                    float(self.own_loggers['train_openings']),
                ), flush=True)

                print("Upd[{}|{}|{}|{:.2f}%]: {}, position-gate: {:.2e}|{:.2e}, gestalt-gate: {:.2e}|{:.2e}".format(
                    self.trainer.local_rank,
                    self.trainer.global_step,
                    self.trainer.current_epoch,
                    (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                    itpersec,
                    float(self.own_loggers['train_prediction_position_gate']),
                    float(self.own_loggers['train_prediction_position_gate_openings']),
                    float(self.own_loggers['train_prediction_gestalt_gate']),
                    float(self.own_loggers['train_prediction_gestalt_gate_openings']),
                ), flush=True)

        self.val_metrics = {}
        return loss

    def compute_ari(self, mask_true, mask_pred):
        """
        Code from https://github.com/jinyangyuan/air-unofficial/tree/59ce7adc023be8876a0045b0c53982498fbfaf1e
        """
        def comb2(x):
            x = x * (x - 1)
            if x.dim() > 1:
                x = x.sum([*range(1, x.dim())])
            return x
        num_pixels = mask_true.sum([*range(1, mask_true.dim())])
        """
        mask_true = mask_true.reshape(
            [mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[-2] * mask_true.shape[-1]])
        mask_pred = mask_pred.reshape(
            [mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[-2] * mask_pred.shape[-1]])
        mat = (mask_true * mask_pred).sum(-1)
        """

        B, M, H, W = mask_true.shape
        N = mask_pred.shape[1]

        # Initialize an empty tensor to store the result
        mat = th.zeros((B, M, N), dtype=mask_true.dtype, device=mask_true.device)

        for b in range(B):
            for m in range(M):
                for n in range(N):
                    # Element-wise multiplication and summation
                    mat[b, m, n] = (mask_true[b, m] * mask_pred[b, n]).sum()



        sum_row = mat.sum(1)
        sum_col = mat.sum(2)
        comb_mat = comb2(mat)
        comb_row = comb2(sum_row)
        comb_col = comb2(sum_col)
        comb_num = comb2(num_pixels)
        comb_prod = (comb_row * comb_col) / comb_num
        comb_mean = 0.5 * (comb_row + comb_col)
        diff = comb_mean - comb_prod
        score = (comb_mat - comb_prod) / diff
        invalid = ((comb_num == 0) + (diff == 0)) > 0
        score = th.where(invalid, th.ones_like(score), score)
        return score

    def compute_ami(self, pred_masks, gt_masks):
        """
        Compute AMI between batches of predicted and ground truth masks.
        
        Parameters:
        - pred_masks: numpy array of shape (B, N, H, W)
        - gt_masks: numpy array of shape (B, M, H, W)
        
        Returns:
        - ami_scores: numpy array of shape (B,) containing AMI scores for each batch
        """
        B = pred_masks.shape[0]  # Number of batches
        ami_scores = np.zeros(B)  # To store AMI scores for each batch
        
        if self.cfg.evalualte_ami:
            for b in range(B):
                # Get predicted and ground truth masks for the b-th batch
                pred = pred_masks[b].detach().cpu().numpy()
                gt = gt_masks[b].detach().cpu().numpy()
            
                valid_gt = reduce(gt, 'b h w -> (h w)', 'max') == 1
                valid_pred = reduce(pred, 'b h w -> (h w)', 'max') == 1
                valid = valid_gt
                
                pred_flat = np.argmax(rearrange(pred, 'n h w -> n (h w)'), axis=0)
                gt_flat = np.argmax(rearrange(gt, 'm h w -> m (h w)'), axis=0)
                
                ami_scores[b] = adjusted_mutual_info_score(gt_flat[valid], pred_flat[valid], average_method='arithmetic')
            
        return th.from_numpy(ami_scores).to(pred_masks.device)

    def validation_step(self, batch, batch_idx):

        if self.gt_mask_seq is None:
            self.log('val_reconstruction_seq_fg_ari', 0)
            self.log('val_prediction_seq_fg_ari', 0)
            self.log('val_reconstruction_seq_ari', 0) 
            self.log('val_prediction_seq_ari', 0)
            self.log('val_seq_ari_sum', 1e-16)

        time_steps     = batch[2]
        fg_mask        = (batch[4] > 0.8).float() 
        instance_masks = (batch[5] > 0.4).float() if "amodal" in self.cfg.model and self.cfg.model.amodal else (batch[5] > 0.8).float() # overlapped or hidden regions of an object are 0.5 in the gt_mask
        gt_masks       = (batch[5] > 0.8).float()

        has_fg_mask = (reduce(fg_mask, 'b t c h w -> b', 'min') >= 0).float()

        if time_steps[0,0].item() != -self.teacher_forcing:
            time_steps     = th.cat((self.last_time_step, time_steps), dim=1)
            fg_mask        = th.cat((self.last_fg_mask, fg_mask), dim=1)
            instance_masks = th.cat((self.last_instance_masks, instance_masks), dim=1)
            gt_masks       = th.cat((self.last_gt_masks, gt_masks), dim=1)

        self.last_fg_mask        = fg_mask[:,-1:]
        self.last_instance_masks = instance_masks[:,-1:]
        self.last_gt_masks       = gt_masks[:,-1:]

        if self.cfg.sequence_len == 1:
            instance_masks = th.cat([instance_masks for _ in range(self.cfg.backprop_steps+1)], dim=1)
            gt_masks       = th.cat([gt_masks for _ in range(self.cfg.backprop_steps+1)], dim=1)
            fg_mask        = th.cat([fg_mask for _ in range(self.cfg.backprop_steps+1)], dim=1)
            input_rgb      = th.cat([batch[0] for _ in range(self.cfg.backprop_steps+1)], dim=1)
            input_depth    = th.cat([batch[1] for _ in range(self.cfg.backprop_steps+1)], dim=1)

            loss, outputs = self.model_step([input_rgb, input_depth, time_steps, batch[3]], batch_idx, prefix='val')
        else:
            loss, outputs = self.model_step(batch[:4], batch_idx, prefix='val')
        
        mask_key = "mask"

        gt_masks_list  = []
        rec_mask_list  = []
        pred_mask_list = []

        for t in range(len(outputs)):
            if time_steps[0,t] >= 0:
                gt_masks_list.append(th.cat((gt_masks[:,t], 1 - fg_mask[:,t]), dim=1))
                rec_mask_list.append((outputs[t]['reconstruction']['mask'] > 0.8).float())
                pred_mask_list.append((outputs[t]['prediction']['mask'] > 0.8).float())


        if time_steps[0,-1].item() >= 0: 

            if time_steps[0,0].item() <= 0:
                self.gt_mask_seq   = th.cat(gt_masks_list, dim=-1).detach()
                self.pred_mask_seq = th.cat(pred_mask_list, dim=-1).detach()
                self.rec_mask_seq  = th.cat(rec_mask_list, dim=-1).detach()
            else:
                # cat horizontal
                self.gt_mask_seq   = th.cat((self.gt_mask_seq,   th.cat(gt_masks_list, dim=-1)), dim=-1).detach()
                self.pred_mask_seq = th.cat((self.pred_mask_seq, th.cat(pred_mask_list, dim=-1)), dim=-1).detach()
                self.rec_mask_seq  = th.cat((self.rec_mask_seq,  th.cat(rec_mask_list, dim=-1)), dim=-1).detach()

            if time_steps[0,-1].item() == self.cfg.sequence_len-1:

                reconstruction_fg_ari = self.compute_ari(self.gt_mask_seq[:,:-1], self.rec_mask_seq[:,:-1])
                prediction_fg_ari     = self.compute_ari(self.gt_mask_seq[:,:-1], self.pred_mask_seq[:,:-1])
                
                reconstuction_ari = self.compute_ari(self.gt_mask_seq, self.rec_mask_seq)
                prediction_ari    = self.compute_ari(self.gt_mask_seq, self.pred_mask_seq)
                
                self.log('val_reconstruction_seq_fg_ari', reconstruction_fg_ari.sum())
                self.log('val_prediction_seq_fg_ari', prediction_fg_ari.sum())
                self.log('val_reconstruction_seq_ari', reconstuction_ari.sum())
                self.log('val_prediction_seq_ari', prediction_ari.sum())
                self.log('val_seq_ari_sum', reconstuction_ari.numel())



        if time_steps[0,-1].item() >= 0 or self.cfg.sequence_len == 1: 

            reconstruction_IoU = 0
            reconstruction_sum = 0
            mean_reconstruction_IoU = 0
            mean_reconstruction_sum = 0
            prediction_IoU = 0
            prediction_sum = 0
            mean_prediction_IoU = 0
            mean_prediction_sum = 0
            reconstruction_F1 = 0
            mean_reconstruction_F1 = 0
            prediction_F1 = 0
            mean_prediction_F1 = 0


            t_start = len(outputs) -1 if self.cfg.sequence_len == 1 else 0
            for t in range(t_start, len(outputs)):
                if time_steps[0,t] >= 0 or self.cfg.sequence_len == 1:

                    rec_num_slots = reduce((outputs[t]['reconstruction'][mask_key][:,:-1] > 0.8).float(), 'b n h w -> b n', 'max')
                    rec_num_slots = reduce(rec_num_slots, 'b n -> b', 'sum')

                    gt_num_slots = reduce((gt_masks[:,t] > 0.8).float(), 'b n h w -> b n', 'max')
                    gt_num_slots = reduce(gt_num_slots, 'b n -> b', 'sum')

                    self.log('val_reconstruction_object_counting', th.sum((rec_num_slots == gt_num_slots).float()))
                    self.log('val_reconstruction_object_counting_sum', rec_num_slots.numel())

                    pred_num_slots = reduce((outputs[t]['prediction'][mask_key][:,:-1] > 0.8).float(), 'b n h w -> b n', 'max')
                    pred_num_slots = reduce(pred_num_slots, 'b n -> b', 'sum')

                    gt_num_slots = reduce((gt_masks[:,t+1] > 0.8).float(), 'b n h w -> b n', 'max')
                    gt_num_slots = reduce(gt_num_slots, 'b n -> b', 'sum')

                    self.log('val_prediction_object_counting', th.sum((pred_num_slots == gt_num_slots).float()))
                    self.log('val_prediction_object_counting_sum', pred_num_slots.numel())


                    # TODO reorganize: here we use 0.8 threshold since ari is currently always (not amdoal) segmentation
                    reconstruction_fg_ari = self.compute_ari((gt_masks[:,t] > 0.8).float(), (outputs[t]['reconstruction'][mask_key][:,:-1] > 0.8).float())
                    prediction_fg_ari     = self.compute_ari((gt_masks[:,t+1] > 0.8).float(), (outputs[t]['prediction'][mask_key][:,:-1] > 0.8).float())
                    
                    reconstuction_ari = self.compute_ari(th.cat(((gt_masks[:,t] > 0.8).float(), 1 - fg_mask[:,t]), dim=1), (outputs[t]['reconstruction'][mask_key] > 0.8).float())
                    prediction_ari    = self.compute_ari(th.cat(((gt_masks[:,t+1] > 0.8).float(), 1 - fg_mask[:,t+1]), dim=1), (outputs[t]['prediction'][mask_key] > 0.8).float())
                    
                    self.log('val_reconstruction_fg_ari', reconstruction_fg_ari.sum())
                    self.log('val_prediction_fg_ari', prediction_fg_ari.sum())
                    self.log('val_reconstruction_ari', reconstuction_ari.sum())
                    self.log('val_prediction_ari', prediction_ari.sum())
                    self.log('val_ari_sum', reconstuction_ari.numel())


                    reconstruction_fg_ami = self.compute_ami((gt_masks[:,t] > 0.8).float(), (outputs[t]['reconstruction'][mask_key][:,:-1] > 0.8).float())
                    prediction_fg_ami     = self.compute_ami((gt_masks[:,t+1] > 0.8).float(), (outputs[t]['prediction'][mask_key][:,:-1] > 0.8).float())
                    
                    reconstuction_ami = self.compute_ami(th.cat(((gt_masks[:,t] > 0.8).float(), 1 - fg_mask[:,t]), dim=1), (outputs[t]['reconstruction'][mask_key] > 0.8).float())
                    prediction_ami    = self.compute_ami(th.cat(((gt_masks[:,t+1] > 0.8).float(), 1 - fg_mask[:,t+1]), dim=1), (outputs[t]['prediction'][mask_key] > 0.8).float())
                    
                    self.log('val_reconstruction_fg_ami', reconstruction_fg_ami.sum())
                    self.log('val_prediction_fg_ami', prediction_fg_ami.sum())
                    self.log('val_reconstruction_ami', reconstuction_ami.sum())
                    self.log('val_prediction_ami', prediction_ami.sum())
                    self.log('val_ami_sum', reconstuction_ami.numel())



                    out_fg_mask  = 1 - outputs[t]['reconstruction'][mask_key][:,-1:]
                    intersection = th.sum(out_fg_mask * fg_mask[:,t], dim=[1,2,3])
                    union        = th.sum(th.maximum(out_fg_mask, fg_mask[:,t]), dim=[1,2,3])
                    reconstruction_IoU += th.sum(intersection / (union + 1e-6))
                    reconstruction_F1  += th.sum(2 * intersection / (th.sum(out_fg_mask + fg_mask[:,t], dim=[1,2,3]) + 1e-6))
                    reconstruction_sum += intersection.shape[0]

                    out_fg_mask  = 1 - outputs[t]['prediction'][mask_key][:,-1:]
                    intersection = th.sum(out_fg_mask * fg_mask[:,t+1], dim=[1,2,3])
                    union        = th.sum(th.maximum(out_fg_mask, fg_mask[:,t+1]), dim=[1,2,3])
                    prediction_IoU += th.sum(intersection / (union + 1e-6))
                    prediction_F1  += th.sum(2 * intersection / (th.sum(out_fg_mask + fg_mask[:,t+1], dim=[1,2,3]) + 1e-6))
                    prediction_sum += intersection.shape[0]

                    # Get the output object masks
                    masks_rec  = (outputs[t]['reconstruction'][mask_key][:, :-1] > 0.8).float() # Shape: [B, N, H, W]
                    masks_pred = (outputs[t]['prediction'][mask_key][:, :-1] > 0.8).float()     # Shape: [B, N, H, W]

                    active_mask_rec  = reduce(masks_rec,  'b n h w -> b n', 'max')
                    active_mask_pred = reduce(masks_pred, 'b n h w -> b n', 'max')

                    target_masks = instance_masks[:, t] # Shape: [B, M, H, W]

                    active_target_masks = reduce((target_masks > 0.8).float(), 'b m h w -> b m', 'max')
                    
                    # Calculate intersection and union for reconstruction
                    intersection_rec = (masks_rec.unsqueeze(2) * target_masks.unsqueeze(1)).sum(dim=(-1, -2))       # Shape: [B, N, M]
                    union_rec = (masks_rec.unsqueeze(2) + target_masks.unsqueeze(1)).clamp_(0, 1).sum(dim=(-1, -2)) # Shape: [B, N, M]
                    
                    # Calculate intersection and union for prediction
                    intersection_pred = (masks_pred.unsqueeze(2) * target_masks.unsqueeze(1)).sum(dim=(-1, -2))       # Shape: [B, N, M]
                    union_pred = (masks_pred.unsqueeze(2) + target_masks.unsqueeze(1)).clamp_(0, 1).sum(dim=(-1, -2)) # Shape: [B, N, M]

                    # Calculate IoU matrices
                    IoU_matrix_rec  = intersection_rec / (union_rec + 1e-6)    # Shape: [B, N, M]
                    IoU_matrix_pred = intersection_pred / (union_pred + 1e-6)  # Shape: [B, N, M]

                    F1_matrix_rec  = 2 * intersection_rec / ((masks_rec.unsqueeze(2) + target_masks.unsqueeze(1)).sum(dim=(-1, -2)) + 1e-6) # Shape: [B, N, M]
                    F1_matrix_pred = 2 * intersection_pred / ((masks_pred.unsqueeze(2) + target_masks.unsqueeze(1)).sum(dim=(-1, -2)) + 1e-6) # Shape: [B, N, M]
                    
                    # For each item in the batch
                    for b in range(masks_rec.shape[0]):
                        if has_fg_mask[b].item() > 0:

                            # Use linear_sum_assignment to find the optimal assignment
                            #row_ind_rec, col_ind_rec   = linear_sum_assignment(-IoU_matrix_rec[b].cpu().numpy()) 
                            #row_ind_pred, col_ind_pred = linear_sum_assignment(-IoU_matrix_pred[b].cpu().numpy())
                            
                            # use intersection here for the review paper
                            row_ind_rec, col_ind_rec   = linear_sum_assignment(-intersection_rec[b].cpu().numpy()) 
                            row_ind_pred, col_ind_pred = linear_sum_assignment(-intersection_pred[b].cpu().numpy())
                            
                            # Calculate mean IoU for the optimal assignment
                            mean_reconstruction_IoU += th.sum(IoU_matrix_rec[b][row_ind_rec, col_ind_rec]    * active_mask_rec[b, row_ind_rec])
                            mean_reconstruction_F1  += th.sum(F1_matrix_rec[b][row_ind_rec, col_ind_rec]     * active_mask_rec[b, row_ind_rec])
                            mean_prediction_IoU     += th.sum(IoU_matrix_pred[b][row_ind_pred, col_ind_pred] * active_mask_pred[b, row_ind_pred])
                            mean_prediction_F1      += th.sum(F1_matrix_pred[b][row_ind_pred, col_ind_pred]  * active_mask_pred[b, row_ind_pred])
                            mean_reconstruction_sum += th.sum(active_target_masks[b])
                            mean_prediction_sum     += th.sum(active_target_masks[b])

            # Log the mean IoUs
            self.log('val_mean_reconstruction_IoU', mean_reconstruction_IoU)
            self.log('val_mean_reconstruction_F1', mean_reconstruction_F1)
            self.log('val_mean_prediction_IoU', mean_prediction_IoU)
            self.log('val_mean_prediction_F1', mean_prediction_F1)
            self.log('val_mean_reconstruction_IoU_sum', mean_reconstruction_sum)
            self.log('val_mean_prediction_IoU_sum', mean_prediction_sum)

            self.log('val_reconstruction_IoU', reconstruction_IoU)
            self.log('val_reconstruction_F1', reconstruction_F1)
            self.log('val_reconstruction_IoU_sum', reconstruction_sum)
            self.log('val_prediction_IoU', prediction_IoU)
            self.log('val_prediction_F1', prediction_F1)
            self.log('val_prediction_IoU_sum', prediction_sum)

            self.log('val_sum', 1)

            print("Test-Enc[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, depth: {:.2e}|{:.2e}|{:.2e}, obj: {:.1f}, binar: {:.3f}, IoU: {:.4f}%|{:.4f}%, F1: {:.4f}%|{:.4f}%, , ARI: {:.4f}|{:.4f}, AMI: {:.4f}|{:.4f}, Seq-ARI: {:.4f}|{:.4f}, OCA: {:.4f}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.val_dataloaders) * 100,
                str(self.timer),
                self.val_metrics['val_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_rgb_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_rgb_l1_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_rgb_ssim_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_depth_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_depth_l1_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_depth_ssim_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_num_slots'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_gestalt_mean'] / self.val_metrics['val_sum'],
                self.val_metrics['val_reconstruction_IoU'] * 100 / self.val_metrics['val_reconstruction_IoU_sum'],
                self.val_metrics['val_mean_reconstruction_IoU'] * 100 / self.val_metrics['val_mean_reconstruction_IoU_sum'],
                self.val_metrics['val_reconstruction_F1'] * 100 / self.val_metrics['val_reconstruction_IoU_sum'],
                self.val_metrics['val_mean_reconstruction_F1'] * 100 / self.val_metrics['val_mean_reconstruction_IoU_sum'],
                self.val_metrics['val_reconstruction_ari'] / self.val_metrics['val_ari_sum'],
                self.val_metrics['val_reconstruction_fg_ari'] / self.val_metrics['val_ari_sum'],
                self.val_metrics['val_reconstruction_ami'] / self.val_metrics['val_ami_sum'],
                self.val_metrics['val_reconstruction_fg_ami'] / self.val_metrics['val_ami_sum'],
                self.val_metrics['val_reconstruction_seq_ari'] / self.val_metrics['val_seq_ari_sum'],
                self.val_metrics['val_reconstruction_seq_fg_ari'] / self.val_metrics['val_seq_ari_sum'],
                self.val_metrics['val_reconstruction_object_counting'] / self.val_metrics['val_reconstruction_object_counting_sum'],
            ), flush=True)

            print("Test-Prd[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, rgb: {:.2e}|{:.2e}|{:.2e}, depth: {:.2e}|{:.2e}|{:.2e}, obj: {:.1f}, gestalt: {:.3f}, Reg: {:.2e}|{:.2e}|{:.2e}, open: {:.2e}, IoU: {:.4f}%|{:.4f}%, F1: {:.4f}%|{:.4f}%, ARI: {:.4f}|{:.4f}, AMI: {:.4f}|{:.4f}, Seq-ARI: {:.4f}|{:.4f}, OCA: {:.4f}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.val_dataloaders) * 100,
                str(self.timer),
                self.val_metrics['val_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_rgb_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_rgb_l1_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_rgb_ssim_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_depth_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_depth_l1_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_depth_ssim_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_num_slots'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_gestalt_mean'] / self.val_metrics['val_sum'],
                self.val_metrics['val_object_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_time_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_position_loss'] / self.val_metrics['val_sum'],
                self.val_metrics['val_openings'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_IoU'] * 100 / self.val_metrics['val_prediction_IoU_sum'],
                self.val_metrics['val_mean_prediction_IoU'] * 100 / self.val_metrics['val_mean_prediction_IoU_sum'],
                self.val_metrics['val_prediction_F1'] * 100 / self.val_metrics['val_prediction_IoU_sum'],
                self.val_metrics['val_mean_prediction_F1'] * 100 / self.val_metrics['val_mean_prediction_IoU_sum'],
                self.val_metrics['val_prediction_ari'] / self.val_metrics['val_ari_sum'],
                self.val_metrics['val_prediction_fg_ari'] / self.val_metrics['val_ari_sum'],
                self.val_metrics['val_prediction_ami'] / self.val_metrics['val_ami_sum'],
                self.val_metrics['val_prediction_fg_ami'] / self.val_metrics['val_ami_sum'],
                self.val_metrics['val_prediction_seq_ari'] / self.val_metrics['val_seq_ari_sum'],
                self.val_metrics['val_prediction_seq_fg_ari'] / self.val_metrics['val_seq_ari_sum'],
                self.val_metrics['val_prediction_object_counting'] / self.val_metrics['val_prediction_object_counting_sum'],
            ), flush=True)

            print("Test-Upd[{}|{}|{}|{:.2f}%]: {}, position-gate: {:.2e}|{:.2e}, gestalt-gate: {:.2e}|{:.2e}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.val_dataloaders) * 100,
                str(self.timer),
                self.val_metrics['val_prediction_position_gate'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_position_gate_openings'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_gestalt_gate'] / self.val_metrics['val_sum'],
                self.val_metrics['val_prediction_gestalt_gate_openings'] / self.val_metrics['val_sum'],
            ), flush=True)

    def test_step(self, batch, batch_idx):
        # Optional: Implement the test step
        pass

    def configure_optimizers(self):
        optimizer = Ranger([
            {'params': self.net.parameters(), 'lr': self.cfg.learning_rate, "weight_decay": self.cfg.weight_decay},
        ])
        return optimizer

