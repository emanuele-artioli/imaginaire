# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""
RAFT-based optical flow network.
Replaces FlowNet2 with torchvision's RAFT implementation.
RAFT is more modern, better performing, and doesn't require custom CUDA extensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from imaginaire.model_utils.fs_vid2vid import resample


class FlowNet(nn.Module):
    """RAFT-based optical flow network.
    
    Drop-in replacement for FlowNet2 with the same interface.
    Uses torchvision's RAFT implementation with pretrained weights.
    
    Args:
        pretrained (bool): Load pretrained weights (default: True)
        fp16 (bool): Use half precision (default: False) - kept for API compatibility
        model_size (str): 'large' or 'small' (default: 'large')
    """
    
    def __init__(self, pretrained=True, fp16=False, model_size='large'):
        super().__init__()
        self.fp16 = fp16
        
        # Load RAFT model
        if model_size == 'large':
            if pretrained:
                weights = Raft_Large_Weights.DEFAULT
                self.flowNet = raft_large(weights=weights)
                print(f"[FlowNet RAFT] Loaded RAFT-Large with pretrained weights")
            else:
                self.flowNet = raft_large(weights=None)
                print(f"[FlowNet RAFT] Loaded RAFT-Large without pretrained weights")
        else:
            if pretrained:
                weights = Raft_Small_Weights.DEFAULT
                self.flowNet = raft_small(weights=weights)
                print(f"[FlowNet RAFT] Loaded RAFT-Small with pretrained weights")
            else:
                self.flowNet = raft_small(weights=None)
                print(f"[FlowNet RAFT] Loaded RAFT-Small without pretrained weights")
        
        # Set to eval mode - we don't train the flow network
        self.flowNet.eval()
        
        # RAFT expects images in [0, 255] range, but imaginaire uses [-1, 1]
        # We'll handle this in preprocessing
        
    def forward(self, input_A, input_B):
        """Compute optical flow between two images.
        
        Args:
            input_A: Source image tensor [B, C, H, W] or [B, N, C, H, W] or [B, T, N, C, H, W]
                     Values in range [-1, 1]
            input_B: Target image tensor, same shape as input_A
            
        Returns:
            flow: Optical flow [B, 2, H, W] (or reshaped for higher dim inputs)
            conf: Confidence map [B, 1, H, W]
        """
        size = input_A.size()
        assert(len(size) == 4 or len(size) == 5 or len(size) == 6)
        
        if len(size) >= 5:
            if len(size) == 5:
                b, n, c, h, w = size
            else:
                b, t, n, c, h, w = size
            input_A = input_A.contiguous().view(-1, c, h, w)
            input_B = input_B.contiguous().view(-1, c, h, w)
            flow, conf = self.compute_flow_and_conf(input_A, input_B)
            if len(size) == 5:
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return flow.view(b, t, n, 2, h, w), conf.view(b, t, n, 1, h, w)
        else:
            return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        """Compute flow and confidence between two images.
        
        Args:
            im1: Source image [B, 3, H, W] in range [-1, 1]
            im2: Target image [B, 3, H, W] in range [-1, 1]
            
        Returns:
            flow: Optical flow [B, 2, H, W]
            conf: Confidence map [B, 1, H, W]
        """
        assert(im1.size()[1] == 3)
        assert(im1.size() == im2.size())
        
        old_h, old_w = im1.size()[2], im1.size()[3]
        
        # RAFT requires dimensions divisible by 8
        new_h = ((old_h - 1) // 8 + 1) * 8
        new_w = ((old_w - 1) // 8 + 1) * 8
        
        if old_h != new_h or old_w != new_w:
            im1_resized = F.interpolate(im1, size=(new_h, new_w), mode='bilinear',
                                        align_corners=False)
            im2_resized = F.interpolate(im2, size=(new_h, new_w), mode='bilinear',
                                        align_corners=False)
        else:
            im1_resized = im1
            im2_resized = im2
        
        # Convert from [-1, 1] to [0, 255] range for RAFT
        im1_raft = (im1_resized + 1.0) * 127.5
        im2_raft = (im2_resized + 1.0) * 127.5
        
        # Compute flow using RAFT
        with torch.no_grad():
            # RAFT returns a list of flow predictions at different iterations
            # We take the last (most refined) prediction
            if self.fp16:
                with torch.amp.autocast("cuda"):
                    flow_predictions = self.flowNet(im1_raft, im2_raft)
            else:
                flow_predictions = self.flowNet(im1_raft, im2_raft)
            
            # Get the final flow prediction
            flow1 = flow_predictions[-1]
        
        # Resize flow back to original size if needed
        if old_h != new_h or old_w != new_w:
            # Scale flow values proportionally
            scale_h = old_h / new_h
            scale_w = old_w / new_w
            flow1 = F.interpolate(flow1, size=(old_h, old_w), mode='bilinear',
                                  align_corners=False)
            flow1[:, 0, :, :] *= scale_w
            flow1[:, 1, :, :] *= scale_h
        
        # Compute confidence based on photometric consistency
        # Warp im2 to im1 using the flow and check reconstruction error
        conf = (self.norm(im1 - resample(im2, flow1)) < 0.02).float()
        
        return flow1, conf

    def norm(self, t):
        """Compute squared L2 norm along channel dimension."""
        return torch.sum(t * t, dim=1, keepdim=True)
