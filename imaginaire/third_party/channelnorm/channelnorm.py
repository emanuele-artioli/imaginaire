# flake8: noqa
import torch
from torch.autograd import Function, Variable
from torch.nn.modules.module import Module

# Try to import CUDA extension, fall back to reference implementation
try:
    import channelnorm_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("[channelnorm] CUDA extension not available, using pure PyTorch implementation")


def channelnorm_ref(input1, norm_deg=2):
    """Pure PyTorch implementation of channel normalization.
    
    Computes the L-p norm across channels for each spatial location.
    
    Args:
        input1: Input tensor [B, C, H, W]
        norm_deg: The degree of the norm (default: 2 for L2 norm)
    
    Returns:
        Channel norm [B, 1, H, W]
    """
    # Compute L-p norm across channel dimension
    if norm_deg == 2:
        output = torch.norm(input1, p=2, dim=1, keepdim=True)
    else:
        output = torch.norm(input1, p=norm_deg, dim=1, keepdim=True)
    return output


class ChannelNormFunction(Function):
    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        ctx.save_for_backward(input1)
        ctx.norm_deg = norm_deg

        if HAS_CUDA_EXT and input1.is_cuda:
            assert input1.is_contiguous()
            b, _, h, w = input1.size()
            output = input1.new(b, 1, h, w).zero_()
            channelnorm_cuda.forward(input1, output, norm_deg)
            ctx.save_for_backward(input1, output)
            return output
        else:
            output = channelnorm_ref(input1, norm_deg)
            ctx.save_for_backward(input1, output)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors
        
        if HAS_CUDA_EXT and input1.is_cuda:
            grad_input1 = Variable(input1.new(input1.size()).zero_())
            channelnorm_cuda.backward(input1, output, grad_output.data,
                                      grad_input1.data, ctx.norm_deg)
            return grad_input1, None
        else:
            # For reference implementation, use autograd
            with torch.enable_grad():
                input1_copy = input1.detach().requires_grad_(True)
                output = channelnorm_ref(input1_copy, ctx.norm_deg)
                output.backward(grad_output)
            return input1_copy.grad, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)
