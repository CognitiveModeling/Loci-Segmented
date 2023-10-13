import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class MemoryEfficientUpscaling1x1Function(Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2, scale_factor):

        # reshape input tensor to 2D
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1

        # SiLU activation function using x * sigmoid(x)
        output2 = output1 * th.sigmoid(output1)

        # Linear patch upscale layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H, W, -1, scale_factor, scale_factor).permute(0, 3, 1, 4, 2, 5)
        output3 = output3.reshape(B, -1, H * scale_factor, W * scale_factor)

        # Save input tensor for backward pass
        ctx.save_for_backward(input, weight1, bias1, weight2)
        ctx.scale_factor = scale_factor

        return output3

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, bias1, weight2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor

        B, C, H, W = grad_output.shape
        grad_output = grad_output.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        grad_output = grad_output.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)

        # Recalculate necessary outputs for backward pass
        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1

        # SiLU activation function using x * sigmoid(x)
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for linear patch upscale layer
        grad_output2 = grad_output
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2   = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)

        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)

        # Gradients for first linear layer
        grad_input = th.matmul(grad_silu, weight1).reshape(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        grad_weight1 = th.matmul(grad_silu.t(), input)
        grad_bias1 = grad_silu.sum(dim=0)

        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None


class MemoryEfficientUpscalingConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2, scale_factor, kernel_size):

        # First layer
        output1 = F.conv2d(input, weight1, bias1, padding=kernel_size // 2)

        # reshape input tensor to 2D
        B, C, H, W = output1.shape
        output1 = output1.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # SiLU activation function using x * sigmoid(x)
        output2 = output1 * th.sigmoid(output1)

        # Linear patch upscale layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H, W, -1, scale_factor, scale_factor).permute(0, 3, 1, 4, 2, 5)
        output3 = output3.reshape(B, -1, H * scale_factor, W * scale_factor)

        # Save input tensor for backward pass
        ctx.save_for_backward(input, weight1, bias1, weight2)
        ctx.scale_factor = scale_factor
        ctx.kernel_size  = kernel_size

        return output3

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, bias1, weight2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        kernel_size  = ctx.kernel_size

        # Recalculate necessary outputs for backward pass
        # First linear layer
        output1 = F.conv2d(input, weight1, bias1, padding=kernel_size // 2)

        # reshape input tensor to 2D
        B, C, H, W = output1.shape
        output1 = output1.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # SiLU activation function using x * sigmoid(x)
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        grad_output = grad_output.view(B, -1, H, scale_factor, W, scale_factor)
        grad_output = grad_output.permute(0, 2, 4, 1, 3, 5).reshape(B * H * W, -1)

        # Gradients for linear patch upscale layer
        grad_output2 = grad_output
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2 = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)

        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)
        grad_silu = grad_silu.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Gradients for first linear layer
        grad_input   = F.grad.conv2d_input(input.shape, weight1, grad_silu, padding=kernel_size // 2)
        grad_weight1 = F.grad.conv2d_weight(input, weight1.shape, grad_silu, padding=kernel_size // 2)
        grad_bias1   = grad_silu.sum(dim=(0, 2, 3))

        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None, None

class MemoryEfficientUpscaling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4, expand_ratio=4, kernel_size=1):
        super(MemoryEfficientUpscaling, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size  = kernel_size
        
        if kernel_size == 1:
            hidden_channels = int(max(out_channels, in_channels) * expand_ratio)

            self.weight1 = nn.Parameter(th.randn(hidden_channels, in_channels))
            self.bias1   = nn.Parameter(th.zeros(hidden_channels))
            self.weight2 = nn.Parameter(th.randn(out_channels * scale_factor * scale_factor, hidden_channels))
            self.bias2   = nn.Parameter(th.zeros(out_channels * scale_factor * scale_factor))
        else:
            self.weight1 = nn.Parameter(th.randn(out_channels * expand_ratio, in_channels, kernel_size, kernel_size))
            self.bias1   = nn.Parameter(th.zeros(out_channels * expand_ratio))
            self.weight2 = nn.Parameter(th.randn(out_channels * scale_factor * scale_factor, out_channels * expand_ratio))
            self.bias2   = nn.Parameter(th.zeros(out_channels * scale_factor * scale_factor))

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, input):
        if self.kernel_size == 1:
            return MemoryEfficientUpscaling1x1Function.apply(input, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor)

        return MemoryEfficientUpscalingConv2dFunction.apply(input, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor, self.kernel_size)
