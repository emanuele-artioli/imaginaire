# flake8: noqa
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
from torch.amp import autocast

# Try to import CUDA extension, fall back to reference implementation
# Note: For PyTorch 2.x, the pure PyTorch fallback is actually preferred as
# the CUDA extension has compatibility issues with newer PyTorch C++ APIs.
try:
    import resample2d_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


def resample2d_ref(input1, input2, kernel_size=1):
    """Pure PyTorch implementation of resample2d (backward warping).
    
    Args:
        input1: Source image [B, C, H1, W1]
        input2: Flow field [B, 2, H2, W2] where [:,0] is x-offset and [:,1] is y-offset
        kernel_size: Not used in reference implementation (defaults to bilinear)
    
    Returns:
        Warped image [B, C, H2, W2]
    """
    b, c, h1, w1 = input1.size()
    _, _, h2, w2 = input2.size()
    
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h2, device=input2.device, dtype=input2.dtype),
        torch.arange(0, w2, device=input2.device, dtype=input2.dtype),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)  # [B, H2, W2]
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)  # [B, H2, W2]
    
    # Add flow to grid
    # Flow is in pixel coordinates, need to convert to [-1, 1] for grid_sample
    x = grid_x + input2[:, 0, :, :]  # x + flow_x
    y = grid_y + input2[:, 1, :, :]  # y + flow_y
    
    # Normalize to [-1, 1]
    x = 2.0 * x / (w1 - 1) - 1.0
    y = 2.0 * y / (h1 - 1) - 1.0
    
    # Stack to create sampling grid [B, H2, W2, 2]
    grid = torch.stack([x, y], dim=-1)
    
    # Sample using bilinear interpolation
    output = F.grid_sample(input1, grid, mode='bilinear', 
                          padding_mode='border', align_corners=True)
    
    return output


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = True

        if HAS_CUDA_EXT and input1.is_cuda:
            assert input1.is_contiguous()
            assert input2.is_contiguous()
            _, d, _, _ = input1.size()
            b, _, h, w = input2.size()
            output = input1.new(b, d, h, w).zero_()
            resample2d_cuda.forward(input1, input2, output, kernel_size)
            return output
        else:
            return resample2d_ref(input1, input2, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        
        if HAS_CUDA_EXT and input1.is_cuda:
            grad_output = grad_output.contiguous()
            assert grad_output.is_contiguous()
            grad_input1 = Variable(input1.new(input1.size()).zero_())
            grad_input2 = Variable(input1.new(input2.size()).zero_())
            resample2d_cuda.backward(input1, input2, grad_output.data,
                                     grad_input1.data, grad_input2.data,
                                     ctx.kernel_size)
            return grad_input1, grad_input2, None
        else:
            # For reference implementation, use autograd
            # This will be slower but works without CUDA extension
            with torch.enable_grad():
                input1_copy = input1.detach().requires_grad_(True)
                input2_copy = input2.detach().requires_grad_(True)
                output = resample2d_ref(input1_copy, input2_copy, ctx.kernel_size)
                output.backward(grad_output)
            return input1_copy.grad, input2_copy.grad, None


class Resample2d(Module):

    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    @autocast(False)
    def forward(self, input1, input2):
        input1, input2 = input1.float(), input2.float()
        input1_c = input1.contiguous()
        # return Resample2dFunction.apply(
        #     input1_c, input2, self.kernel_size, self.bilinear)
        return Resample2dFunction.apply(
            input1_c, input2, self.kernel_size)