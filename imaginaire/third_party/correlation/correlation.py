# flake8: noqa
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Function

# Try to import CUDA extension, fall back to reference implementation
try:
    import correlation_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("[correlation] CUDA extension not available, using pure PyTorch implementation")


def correlation_ref(input1, input2, pad_size, kernel_size, max_displacement, 
                    stride1, stride2, corr_multiply):
    """Pure PyTorch implementation of correlation layer.
    
    Computes the correlation between two feature maps.
    This is used in optical flow estimation networks like FlowNet.
    
    Args:
        input1: First feature map [B, C, H, W]
        input2: Second feature map [B, C, H, W]
        pad_size: Padding size
        kernel_size: Kernel size for local patches
        max_displacement: Maximum displacement for correlation
        stride1: Stride for first input
        stride2: Stride within neighborhood
        corr_multiply: Multiplication factor for correlation
    
    Returns:
        Correlation volume [B, D*D, H', W'] where D = 2*max_displacement/stride2 + 1
    """
    b, c, h, w = input1.size()
    
    # Pad inputs
    if pad_size > 0:
        input1 = F.pad(input1, [pad_size] * 4, mode='constant', value=0)
        input2 = F.pad(input2, [pad_size] * 4, mode='constant', value=0)
    
    _, _, h_padded, w_padded = input1.size()
    
    # Output dimensions
    out_h = (h_padded - 2 * max_displacement - kernel_size) // stride1 + 1
    out_w = (w_padded - 2 * max_displacement - kernel_size) // stride1 + 1
    
    # Number of displacement steps
    d = 2 * max_displacement // stride2 + 1
    
    # Initialize output
    output = input1.new_zeros(b, d * d, out_h, out_w)
    
    # Compute correlation for each displacement
    for i in range(d):
        for j in range(d):
            # Displacement offsets
            dy = (i - d // 2) * stride2
            dx = (j - d // 2) * stride2
            
            # Extract patches from both inputs with displacement
            for y in range(out_h):
                for x in range(out_w):
                    y1 = y * stride1 + max_displacement
                    x1 = x * stride1 + max_displacement
                    
                    y2 = y1 + dy
                    x2 = x1 + dx
                    
                    # Check bounds
                    if 0 <= y2 < h_padded - kernel_size + 1 and 0 <= x2 < w_padded - kernel_size + 1:
                        patch1 = input1[:, :, y1:y1+kernel_size, x1:x1+kernel_size]
                        patch2 = input2[:, :, y2:y2+kernel_size, x2:x2+kernel_size]
                        
                        # Correlation = sum of element-wise product
                        corr = (patch1 * patch2).sum(dim=(1, 2, 3))
                        output[:, i * d + j, y, x] = corr * corr_multiply
    
    return output


def correlation_ref_fast(input1, input2, pad_size, kernel_size, max_displacement,
                         stride1, stride2, corr_multiply):
    """Faster pure PyTorch implementation using unfold operations."""
    b, c, h, w = input1.size()
    
    # Pad inputs
    if pad_size > 0:
        input1 = F.pad(input1, [pad_size] * 4, mode='constant', value=0)
        input2 = F.pad(input2, [pad_size] * 4, mode='constant', value=0)
    
    _, _, h_padded, w_padded = input1.size()
    
    # Output dimensions  
    out_h = (h_padded - 2 * max_displacement - kernel_size) // stride1 + 1
    out_w = (w_padded - 2 * max_displacement - kernel_size) // stride1 + 1
    
    # Number of displacement steps
    d = 2 * max_displacement // stride2 + 1
    
    # Initialize output
    outputs = []
    
    # For each displacement, compute correlation using vectorized ops
    for i in range(d):
        for j in range(d):
            dy = (i - d // 2) * stride2
            dx = (j - d // 2) * stride2
            
            # Shifted coordinates for input2
            y_start = max_displacement + dy
            x_start = max_displacement + dx
            y_end = y_start + out_h * stride1
            x_end = x_start + out_w * stride1
            
            # Check bounds and compute correlation
            if (y_start >= 0 and x_start >= 0 and 
                y_end <= h_padded and x_end <= w_padded):
                
                # Extract regions
                region1 = input1[:, :, max_displacement:max_displacement + out_h * stride1:stride1,
                                      max_displacement:max_displacement + out_w * stride1:stride1]
                region2 = input2[:, :, y_start:y_end:stride1, x_start:x_end:stride1]
                
                # Compute correlation (element-wise product, sum over channels)
                corr = (region1 * region2).sum(dim=1, keepdim=True) * corr_multiply
                outputs.append(corr)
            else:
                outputs.append(input1.new_zeros(b, 1, out_h, out_w))
    
    return torch.cat(outputs, dim=1)


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            corr_multiply,
            input1,
            input2):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        if HAS_CUDA_EXT and input1.is_cuda:
            with torch.cuda.device_of(input1):
                rbot1 = input1.new()
                rbot2 = input2.new()
                output = input1.new()

                correlation_cuda.forward(
                    input1,
                    input2,
                    rbot1,
                    rbot2,
                    output,
                    ctx.pad_size,
                    ctx.kernel_size,
                    ctx.max_displacement,
                    ctx.stride1,
                    ctx.stride2,
                    ctx.corr_multiply)

            return output
        else:
            return correlation_ref_fast(input1, input2, pad_size, kernel_size,
                                       max_displacement, stride1, stride2, corr_multiply)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        if HAS_CUDA_EXT and input1.is_cuda:
            with torch.cuda.device_of(input1):
                rbot1 = input1.new()
                rbot2 = input2.new()

                grad_input1 = input1.new()
                grad_input2 = input2.new()

                correlation_cuda.backward(
                    input1,
                    input2,
                    rbot1,
                    rbot2,
                    grad_output,
                    grad_input1,
                    grad_input2,
                    ctx.pad_size,
                    ctx.kernel_size,
                    ctx.max_displacement,
                    ctx.stride1,
                    ctx.stride2,
                    ctx.corr_multiply)

            return None, None, None, None, None, None, grad_input1, grad_input2
        else:
            # For reference implementation, use autograd
            with torch.enable_grad():
                input1_copy = input1.detach().requires_grad_(True)
                input2_copy = input2.detach().requires_grad_(True)
                output = correlation_ref_fast(input1_copy, input2_copy, ctx.pad_size,
                                             ctx.kernel_size, ctx.max_displacement,
                                             ctx.stride1, ctx.stride2, ctx.corr_multiply)
                output.backward(grad_output)
            return None, None, None, None, None, None, input1_copy.grad, input2_copy.grad

class Correlation(Module):
    def __init__(
            self,
            pad_size=0,
            kernel_size=0,
            max_displacement=0,
            stride1=1,
            stride2=2,
            corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = CorrelationFunction.apply(
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply,
            input1,
            input2)

        return result
