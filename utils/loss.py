import torch as th
import torchvision as tv
from torch import nn
from utils.utils import BatchToSharedObjects, SharedObjectsToBatch, LambdaModule, RGB2YCbCr
from einops import rearrange, repeat, reduce
from utils.optimizers import Ranger
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment

__author__ = "Manuel Traub"

class GestaltLoss(nn.Module):
    def __init__(self, warmup_period=1000.0):
        super(GestaltLoss, self).__init__()
        self.register_buffer('num_updates', th.tensor(0.0))
        self.warmup_period = warmup_period

    def forward(self, gestalt):
        if self.num_updates < 30 * self.warmup_period:
            scaling_factor = max(0.0, min(1, 0.1 ** (self.num_updates.item() / self.warmup_period - 1)))
            loss = th.mean(th.abs(gestalt - 0.5)) * scaling_factor
        else:
            loss = th.tensor(0.0, device=self.num_updates.device)

        self.num_updates.add_(1.0)
        return loss

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, input, target):
        masked     = (target > 0.5).float()
        non_masked = (target <= 0.5).float()

        error = (input - target)**2
        
        masked_loss     = th.mean(th.sum(error * masked, dim=(1,2,3)) / (th.sum(masked, dim=(1,2,3)) + 1e-8))
        non_masked_loss = th.mean(th.sum(error * non_masked, dim=(1,2,3)) / (th.sum(non_masked, dim=(1,2,3)) + 1e-8))
        return masked_loss + non_masked_loss

class DecayingFactor(nn.Module):
    def __init__(self, warmup_period=2500.0, min_factor=0.01, inverse=False):
        super(DecayingFactor, self).__init__()
        self.register_buffer('num_updates', th.tensor(0.0))
        self.warmup_period = warmup_period
        self.min_factor = min_factor
        self.inverse = inverse

    def get(self):
        factor = max(self.min_factor, min(1, 0.1 ** (self.num_updates.item() / self.warmup_period - 1)))
        if self.inverse:
            factor = 1 - factor
        self.num_updates.add_(1.0)
        return factor

    def forward(self, x):
        return x * self.get()

class DecayingMSELoss(nn.Module):
    def __init__(self, warmup_period=2500.0, min_factor=0.01):
        super(DecayingMSELoss, self).__init__()
        self.register_buffer('num_updates', th.tensor(0.0))
        self.warmup_period = warmup_period
        self.min_factor = min_factor

    def forward(self, pred, target):
        scaling_factor = max(self.min_factor, min(1, 0.1 ** (self.num_updates.item() / self.warmup_period - 1)))
        loss = th.mean((pred - target)**2) * scaling_factor

        self.num_updates.add_(1.0)
        return loss

class PositionLoss(nn.Module):
    def __init__(self, num_slots: int, teacher_forcing: int):
        super(PositionLoss, self).__init__()

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_slots))
        self.last_mask = None
        self.t = 0 
        self.teacher_forcing = teacher_forcing

    def reset_state(self):
        self.last_mask = None
        self.t = 0

    def forward(self, position, position_last, mask):
        
        mask = th.max(th.max(mask, dim=3)[0], dim=2)[0]
        mask = self.to_batch(mask).detach()
        self.t = self.t + 1

        if self.last_mask is None or self.t <= self.teacher_forcing:
            self.last_mask = mask.detach()
            return th.zeros(1, device=mask.device)

        self.last_mask = th.maximum(self.last_mask, mask)

        position      = self.to_batch(position)
        position_last = self.to_batch(position_last).detach()

        position      = th.cat((position[:,:2], 0.25 * th.sigmoid(position[:,3:4])), dim=1) 
        position_last = th.cat((position_last[:,:2], 0.25 * th.sigmoid(position_last[:,3:4])), dim=1) 

        return 0.01 * th.mean(self.last_mask * (position - position_last)**2)


class MaskModulatedObjectLoss(nn.Module):
    def __init__(self, num_slots: int, teacher_forcing: int):
        super(MaskModulatedObjectLoss, self).__init__()

        self.to_batch  = SharedObjectsToBatch(num_slots)
        self.last_mask = None
        self.t = 0 
        self.teacher_forcing = teacher_forcing

    def reset_state(self):
        self.last_mask = None
        self.t = 0
    
    def forward(
        self, 
        object_output,
        object_target,
        mask: th.Tensor
    ):
        mask = self.to_batch(mask).detach()
        mask = th.max(th.max(mask, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        self.t = self.t + 1

        if self.last_mask is None or self.t <= self.teacher_forcing:
            self.last_mask = mask.detach()
            return th.zeros(1, device=mask.device)

        self.last_mask = th.maximum(self.last_mask, mask).detach()

        object_output = th.sigmoid(self.to_batch(object_output) - 2.5)
        object_target = th.sigmoid(self.to_batch(object_target) - 2.5)

        return th.mean((1 - mask) * self.last_mask * (object_output - object_target)**2)

class ObjectModulator(nn.Module):
    def __init__(self, num_slots: int): 
        super(ObjectModulator, self).__init__()
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_slots))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_slots))
        self.position  = None
        self.gestalt   = None

    def reset_state(self):
        self.position = None
        self.gestalt  = None

    def forward(self, position: th.Tensor, gestalt: th.Tensor, mask: th.Tensor):

        position = self.to_batch(position)
        gestalt  = self.to_batch(gestalt)

        if self.position is None or self.gestalt is None:
            self.position = position.detach()
            self.gestalt  = gestalt.detach()
            return self.to_shared(position), self.to_shared(gestalt)

        mask = th.max(th.max(mask, dim=3)[0], dim=2)[0]
        mask = self.to_batch(mask.detach())

        _position = mask * position + (1 - mask) * self.position
        position  = th.cat((position[:,:-1], _position[:,-1:]), dim=1)
        gestalt   = mask * gestalt  + (1 - mask) * self.gestalt

        self.gestalt = gestalt.detach()
        self.position = position.detach()
        return self.to_shared(position), self.to_shared(gestalt)

class MoveToCenter(nn.Module):
    def __init__(self, num_slots: int):
        super(MoveToCenter, self).__init__()

        self.to_batch2d = SharedObjectsToBatch(num_slots)
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_slots))
    
    def forward(self, input: th.Tensor, position: th.Tensor):
        
        input    = self.to_batch2d(input)
        position = self.to_batch(position).detach()
        position = th.stack((position[:,0], position[:,1]), dim=1)

        theta = th.tensor([1, 0, 0, 1], dtype=th.float, device=input.device).view(1,2,2)
        theta = repeat(theta, '1 a b -> n a b', n=input.shape[0])

        position = rearrange(position, 'b c -> b c 1')
        theta    = th.cat((theta, position), dim=2)

        grid   = nn.functional.affine_grid(theta, input.shape, align_corners=False)
        output = nn.functional.grid_sample(input, grid, align_corners=False)

        return output

class TranslationInvariantObjectLoss(nn.Module):
    def __init__(self, num_slots: int, teacher_forcing: int):
        super(TranslationInvariantObjectLoss, self).__init__()

        self.move_to_center  = MoveToCenter(num_slots)
        self.to_batch        = SharedObjectsToBatch(num_slots)
        self.last_mask       = None
        self.t               = 0 
        self.teacher_forcing = teacher_forcing

    def reset_state(self):
        self.last_mask = None
        self.t = 0
    
    def forward(
        self, 
        mask: th.Tensor,
        object1: th.Tensor, 
        position1: th.Tensor,
        object2: th.Tensor, 
        position2: th.Tensor,
    ):
        mask = self.to_batch(mask).detach()
        mask = th.max(th.max(mask, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        self.t = self.t + 1

        if self.last_mask is None or self.t <= self.teacher_forcing:
            self.last_mask = mask.detach()
            return th.zeros(1, device=mask.device)

        self.last_mask = th.maximum(self.last_mask, mask).detach()

        object1 = self.move_to_center(th.sigmoid(object1 - 2.5), position1)
        object2 = self.move_to_center(th.sigmoid(object2 - 2.5), position2)

        return th.mean(self.last_mask * (object1 - object2)**2)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return th.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)  

class MaskedL1SSIMLoss(nn.Module):
    def __init__(self, ssim_factor = 0.85):
        super(MaskedL1SSIMLoss, self).__init__()

        self.ssim = SSIM()
        self.ssim_factor = ssim_factor

    def forward(self, output, target, mask):
        
        l1    = th.abs(output - target) * mask
        ssim  = self.ssim(output, target) * mask

        numel = th.sum(mask, dim=(1, 2, 3)) + 1e-7

        l1 = th.sum(l1, dim=(1, 2, 3)) / numel
        ssim = th.sum(ssim, dim=(1, 2, 3)) / numel

        f = self.ssim_factor

        return th.mean(l1 * (1 - f) + ssim * f), th.mean(l1), th.mean(ssim)

class L1SSIMLoss(nn.Module):
    def __init__(self, ssim_factor = 0.85):
        super(L1SSIMLoss, self).__init__()

        self.ssim = SSIM()
        self.ssim_factor = ssim_factor

    def forward(self, output, target):
        
        l1    = th.abs(output - target)
        ssim  = self.ssim(output, target)

        f = self.ssim_factor

        return th.mean(l1 * (1 - f) + ssim * f), th.mean(l1), th.mean(ssim)

class RGBL1SSIMLoss(nn.Module):
    def __init__(self, ssim_factor = 0.5):
        super(RGBL1SSIMLoss, self).__init__()

        self.ssim = SSIM()
        self.ssim_factor = ssim_factor

    def forward(self, output, target):

        grey_output = output[:, 0:1] * 0.299 + output[:, 1:2] * 0.587 + output[:, 2:3] * 0.114
        grey_target = target[:, 0:1] * 0.299 + target[:, 1:2] * 0.587 + target[:, 2:3] * 0.114
        
        l1    = (output - target)**2
        ssim  = self.ssim(grey_output, grey_target)

        f = self.ssim_factor

        return th.mean(l1 * (1 - f) + ssim * f), th.mean(l1), th.mean(ssim)

class YCbCrL2SSIMLoss(nn.Module):
    def __init__(self):
        super(YCbCrL2SSIMLoss, self).__init__()

        self.to_YCbCr = RGB2YCbCr()
        self.ssim     = SSIM()
    
    def forward(self, x, y):

        x = self.to_YCbCr(x)

        with th.no_grad():
            y = self.to_YCbCr(y.detach()).detach()

        y_loss  = th.mean(self.ssim(x[:,0:1], y[:,0:1]))
        cb_loss = th.mean((x[:,1] - y[:,1])**2) * 10
        cr_loss = th.mean((x[:,2] - y[:,2])**2) * 10

        loss = y_loss + cb_loss + cr_loss
        return loss, cb_loss + cr_loss, y_loss

class WeightedYCbCrL2SSIMLoss(nn.Module):
    def __init__(self):
        super(WeightedYCbCrL2SSIMLoss, self).__init__()

        self.to_YCbCr = RGB2YCbCr()
        self.ssim     = SSIM()
    
    def forward(self, x, y, w):

        if self.rgb_input:
            x = self.to_YCbCr(x)

            with th.no_grad():
                y = self.to_YCbCr(y.detach()).detach()

        y_loss  = th.mean(w * self.ssim(x[:,0:1], y[:,0:1]))
        cb_loss = th.mean(w * (x[:,1] - y[:,1])**2) * 10
        cr_loss = th.mean(w * (x[:,2] - y[:,2])**2) * 10

        loss = y_loss + cb_loss + cr_loss
        return loss, cb_loss + cr_loss, y_loss

class MaskedYCbCrL2SSIMLoss(nn.Module):
    def __init__(self, rgb_input = True):
        super(MaskedYCbCrL2SSIMLoss, self).__init__()
        
        self.rgb_input = rgb_input
        self.to_YCbCr  = RGB2YCbCr()
        self.ssim      = SSIM()
    
    def forward(self, x, y, mask):

        if self.rgb_input:
            x = self.to_YCbCr(x)

            with th.no_grad():
                y = self.to_YCbCr(y.detach()).detach()

        y_loss  = self.ssim(x[:,0:1], y[:,0:1]) * mask
        cb_loss = (x[:,1:2] - y[:,1:2])**2 * 10 * mask
        cr_loss = (x[:,2:3] - y[:,2:3])**2 * 10 * mask

        numel = th.sum(mask, dim=(1, 2, 3)) + 1e-7

        y_loss  = th.sum(y_loss,  dim=(1, 2, 3)) / numel
        cb_loss = th.sum(cb_loss, dim=(1, 2, 3)) / numel
        cr_loss = th.sum(cr_loss, dim=(1, 2, 3)) / numel

        loss = y_loss + cb_loss + cr_loss
        return th.mean(loss), th.mean(cb_loss + cr_loss), th.mean(y_loss)

class UncertaintyYCbCrL2SSIMLoss(nn.Module):
    def __init__(self):
        super(UncertaintyYCbCrL2SSIMLoss, self).__init__()

        self.to_YCbCr = RGB2YCbCr()
        self.ssim     = SSIM()
    
    def forward(self, x, y, uncertainty):

        with th.no_grad():
            x = self.to_YCbCr(x.detach()).detach()
            y = self.to_YCbCr(y.detach()).detach()

        y_loss  = self.ssim(x[:,0:1], y[:,0:1]) * uncertainty
        cb_loss = (x[:,1:2] - y[:,1:2])**2 * 10 * uncertainty
        cr_loss = (x[:,2:3] - y[:,2:3])**2 * 10 * uncertainty

        return y_loss + cb_loss + cr_loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, output, target, mask):
        mse   = ((output - target)**2) * mask
        numel = th.sum(mask, dim=(1, 2, 3)) + 1e-7

        return th.mean(th.sum(mse, dim=(1, 2, 3)) / numel)



class YCbCrL1SSIMLoss(nn.Module):
    def __init__(self, factor_Y = 0.95, factor_Cb = 0.025, factor_Cr = 0.025):
        super(YCbCrL1SSIMLoss, self).__init__()
        self.factor_Y  = factor_Y
        self.factor_Cb = factor_Cb
        self.factor_Cr = factor_Cr

        self.to_YCbCr = RGB2YCbCr()
        self.l1ssim   = L1SSIMLoss()
    
    def forward(self, x, y):
        x = self.to_YCbCr(x)

        with th.no_grad():
            y = self.to_YCbCr(y.detach()).detach()

        y_loss, l1, ssim = self.l1ssim(x[:,0:1], y[:,0:1])

        cb_loss = th.mean((x[:,1] - y[:,1])**2)
        cr_loss = th.mean((x[:,2] - y[:,2])**2)

        cr_factor = (y_loss / cr_loss).detach() * self.factor_Cr
        cb_factor = (y_loss / cb_loss).detach() * self.factor_Cb
        
        sum_factors = cr_factor + cb_factor + self.factor_Y

        y_factor  = self.factor_Y  / sum_factors
        cb_factor = cb_factor / sum_factors
        cr_factor = cr_factor / sum_factors

        loss = y_loss * y_factor + cb_loss * cb_factor + cr_loss * cr_factor

        return loss, l1, ssim
