"""
Custom dataset loader for tennis pose transfer
Loads paired videos from directory structure:
  dataset/
    match_name/
      sequence/
        crops/
          person_id/
            00000.png, 00001.png, ...
        poses/
          person_id/
            00000.png, 00001.png, ...
"""

import os
import cv2
import random
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import numpy as np

from imaginaire.utils.distributed import master_only_print as print


class Dataset(TorchDataset):
    r"""Custom dataset for tennis pose transfer using fs_vid2vid.
    
    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
        sequence_length (int): Length of sequences to output.
    """
    
    def __init__(self, cfg, is_inference=False, sequence_length=None, is_test=False):
        self.cfg = cfg
        self.is_inference = is_inference
        self.is_test = is_test
        self.root = cfg.data.train.roots[0] if not is_test else cfg.data.val.roots[0]
        
        # Set sequence length
        if sequence_length is None:
            if is_inference or is_test:
                self.sequence_length = cfg.data.val.augmentations.get('sequence_length', 4)
            else:
                self.sequence_length = cfg.data.train.initial_sequence_length
        else:
            self.sequence_length = sequence_length
        
        self.few_shot_K = cfg.data.initial_few_shot_K
        
        # Parse output_h_w
        if hasattr(cfg.data, 'output_h_w'):
            output_h_w = cfg.data.output_h_w
            if isinstance(output_h_w, str):
                self.output_h_w = tuple(map(int, output_h_w.split(',')))
            else:
                self.output_h_w = output_h_w
        else:
            self.output_h_w = (512, 512)
        
        # Build index of all available sequences
        self.sequences = []
        self._build_index()
        
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {self.root}")
        
        # Calculate maximum sequence length available
        self.sequence_length_max = max([s['num_frames'] for s in self.sequences])
        
        print(f"Found {len(self.sequences)} sequences")
        print(f"Max sequence length available: {self.sequence_length_max}")
        
    def _build_index(self):
        """Build index of all available (match, sequence, person) triplets"""
        self.sequences = []
        
        # Iterate through match directories
        for match_dir in sorted(os.listdir(self.root)):
            match_path = os.path.join(self.root, match_dir)
            if not os.path.isdir(match_path):
                continue
            
            # Iterate through sequence directories (e.g., 012, 015, etc.)
            for seq_dir in sorted(os.listdir(match_path)):
                seq_path = os.path.join(match_path, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                
                crops_path = os.path.join(seq_path, 'crops')
                poses_path = os.path.join(seq_path, 'poses')
                
                if not os.path.isdir(crops_path) or not os.path.isdir(poses_path):
                    continue
                
                # Iterate through person directories
                for person_id in sorted(os.listdir(crops_path)):
                    person_crops = os.path.join(crops_path, person_id)
                    person_poses = os.path.join(poses_path, person_id)
                    
                    if not os.path.isdir(person_crops) or not os.path.isdir(person_poses):
                        continue
                    
                    # Get all frames
                    crop_frames = sorted([f for f in os.listdir(person_crops) if f.endswith('.png')])
                    pose_frames = sorted([f for f in os.listdir(person_poses) if f.endswith('.png')])
                    
                    num_frames = min(len(crop_frames), len(pose_frames))
                    
                    # Need at least sequence_length + few_shot_K frames
                    if num_frames >= (self.sequence_length + self.few_shot_K):
                        self.sequences.append({
                            'match': match_dir,
                            'sequence': seq_dir,
                            'person': person_id,
                            'crop_dir': person_crops,
                            'pose_dir': person_poses,
                            'num_frames': num_frames
                        })
    
    def _load_frame(self, image_path):
        """Load and normalize image frame"""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read image: {image_path}")
        
        # Convert BGR to RGB and normalize to [-1, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # Resize if needed
        if img.shape[:2] != self.output_h_w[::-1]:
            img = cv2.resize(img, (self.output_h_w[1], self.output_h_w[0]))
        
        return img
    
    def set_sequence_length(self, sequence_length, few_shot_K=None):
        """Update sequence length dynamically"""
        self.sequence_length = sequence_length
        if few_shot_K is not None:
            self.few_shot_K = few_shot_K
        
        # Rebuild index with new constraints
        old_sequences = self.sequences
        self.sequences = []
        self._build_index()
        
        print(f'Updated sequence length to {sequence_length}, few_shot_K to {self.few_shot_K}')
        print(f'Available sequences: {len(self.sequences)} (was {len(old_sequences)})')

    def num_inference_sequences(self):
        """Return the number of sequences for inference"""
        return len(self.sequences)
    
    def set_inference_sequence_idx(self, driving_idx, few_shot_seq_idx=None, few_shot_frame_idx=0):
        """Set the current sequence for inference iteration
        
        Args:
            driving_idx: Index of the driving (target pose) sequence
            few_shot_seq_idx: Index of the few-shot reference sequence (default: same as driving)
            few_shot_frame_idx: Frame index within the few-shot sequence to use
        """
        if few_shot_seq_idx is None:
            few_shot_seq_idx = driving_idx
            
        self.inference_sequence_idx = driving_idx
        self.few_shot_sequence_idx = few_shot_seq_idx
        self.few_shot_frame_idx = few_shot_frame_idx
        
        # In inference mode, we'll iterate over all frames of this sequence
        seq_info = self.sequences[driving_idx]
        few_shot_seq_info = self.sequences[few_shot_seq_idx]
        
        self.inference_frames = []
        
        crop_frames = sorted([f for f in os.listdir(seq_info['crop_dir']) if f.endswith('.png')])
        pose_frames = sorted([f for f in os.listdir(seq_info['pose_dir']) if f.endswith('.png')])
        
        # Few-shot frames from the reference sequence
        fs_crop_frames = sorted([f for f in os.listdir(few_shot_seq_info['crop_dir']) if f.endswith('.png')])
        fs_pose_frames = sorted([f for f in os.listdir(few_shot_seq_info['pose_dir']) if f.endswith('.png')])
        
        # Create list of all frame pairs for this sequence
        num_frames = min(len(crop_frames), len(pose_frames))
        for frame_idx in range(num_frames):
            self.inference_frames.append({
                'seq_info': seq_info,
                'frame_idx': frame_idx,
                'crop_frames': crop_frames,
                'pose_frames': pose_frames,
                'fs_seq_info': few_shot_seq_info,
                'fs_crop_frames': fs_crop_frames,
                'fs_pose_frames': fs_pose_frames,
            })
        
        # Store few-shot frame index for reference
        self.inference_few_shot_frame_idx = few_shot_frame_idx
    
    def __len__(self):
        if self.is_inference and hasattr(self, 'inference_frames'):
            return len(self.inference_frames)
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.is_inference and hasattr(self, 'inference_frames'):
            return self._get_inference_item(idx)
        return self._get_train_item(idx)
    
    def _get_inference_item(self, idx):
        """Get a single frame for inference"""
        frame_info = self.inference_frames[idx]
        seq_info = frame_info['seq_info']
        frame_idx = frame_info['frame_idx']
        crop_frames = frame_info['crop_frames']
        pose_frames = frame_info['pose_frames']
        
        # Few-shot info
        fs_seq_info = frame_info['fs_seq_info']
        fs_crop_frames = frame_info['fs_crop_frames']
        fs_pose_frames = frame_info['fs_pose_frames']
        
        # Load current frame (driving pose)
        crop_path = os.path.join(seq_info['crop_dir'], crop_frames[frame_idx])
        pose_path = os.path.join(seq_info['pose_dir'], pose_frames[frame_idx])
        
        crop_img = self._load_frame(crop_path)
        pose_img = self._load_frame(pose_path)
        
        driving_images = torch.from_numpy(crop_img.transpose(2, 0, 1)).float().unsqueeze(0)  # (1, C, H, W)
        driving_poses = torch.from_numpy(pose_img.transpose(2, 0, 1)).float().unsqueeze(0)
        
        # Load few-shot frames (from reference sequence)
        few_shot_images = []
        few_shot_poses = []
        
        # Use the specified few-shot frame index
        fs_idx = min(self.inference_few_shot_frame_idx, len(fs_crop_frames) - 1)
        for k in range(self.few_shot_K):
            fs_frame_idx = fs_idx + k
            if fs_frame_idx >= len(fs_crop_frames):
                fs_frame_idx = len(fs_crop_frames) - 1
                
            fs_crop_path = os.path.join(fs_seq_info['crop_dir'], fs_crop_frames[fs_frame_idx])
            fs_pose_path = os.path.join(fs_seq_info['pose_dir'], fs_pose_frames[fs_frame_idx])
            
            fs_crop = self._load_frame(fs_crop_path)
            fs_pose = self._load_frame(fs_pose_path)
            
            few_shot_images.append(torch.from_numpy(fs_crop.transpose(2, 0, 1)).float())
            few_shot_poses.append(torch.from_numpy(fs_pose.transpose(2, 0, 1)).float())
        
        few_shot_images = torch.stack(few_shot_images, dim=0)  # (K, C, H, W)
        few_shot_poses = torch.stack(few_shot_poses, dim=0)
        
        # Create key for output naming - needs to match expected format after batching
        # After DataLoader batching: data['key']['images'][0][0] should be a string
        # DataLoader collates strings into tuples, so we use a single string
        key = f"{seq_info['match']}/{seq_info['sequence']}/{seq_info['person']}/{crop_frames[frame_idx]}"
        
        data = {
            'images': driving_images,  # (1, C, H, W) - current frame
            'label': driving_poses,     # (1, C, H, W) - current pose
            'few_shot_images': few_shot_images,  # (K, C, H, W)
            'few_shot_label': few_shot_poses,   # (K, C, H, W)
            'key': {'images': [key]},  # Single-element list, after batching: [0][0] will get the string from tuple
        }
        
        return data
    
    def _get_train_item(self, idx):
        """Get training data (sequence of frames)"""
        seq_info = self.sequences[idx]
        
        # Get list of all frames
        crop_frames = sorted([f for f in os.listdir(seq_info['crop_dir']) if f.endswith('.png')])
        pose_frames = sorted([f for f in os.listdir(seq_info['pose_dir']) if f.endswith('.png')])
        
        # Sample frame indices
        if self.is_test:
            # For testing, use beginning frames
            start_idx = 0
        else:
            # For training, randomly sample
            max_start = seq_info['num_frames'] - (self.sequence_length + self.few_shot_K)
            start_idx = random.randint(0, max_start)
        
        # Get few-shot frames (reference)
        few_shot_indices = list(range(start_idx, start_idx + self.few_shot_K))
        
        # Get driving sequence frames
        driving_start = start_idx + self.few_shot_K
        driving_indices = list(range(driving_start, driving_start + self.sequence_length))
        
        # Load few-shot frames
        few_shot_images = []
        few_shot_poses = []
        
        for frame_idx in few_shot_indices:
            crop_path = os.path.join(seq_info['crop_dir'], crop_frames[frame_idx])
            pose_path = os.path.join(seq_info['pose_dir'], pose_frames[frame_idx])
            
            crop_img = self._load_frame(crop_path)
            pose_img = self._load_frame(pose_path)
            
            few_shot_images.append(torch.from_numpy(crop_img.transpose(2, 0, 1)).float())
            few_shot_poses.append(torch.from_numpy(pose_img.transpose(2, 0, 1)).float())
        
        # Load driving sequence frames
        driving_images = []
        driving_poses = []
        
        for frame_idx in driving_indices:
            crop_path = os.path.join(seq_info['crop_dir'], crop_frames[frame_idx])
            pose_path = os.path.join(seq_info['pose_dir'], pose_frames[frame_idx])
            
            crop_img = self._load_frame(crop_path)
            pose_img = self._load_frame(pose_path)
            
            driving_images.append(torch.from_numpy(crop_img.transpose(2, 0, 1)).float())
            driving_poses.append(torch.from_numpy(pose_img.transpose(2, 0, 1)).float())
        
        # Stack frames along time dimension for driving sequence
        driving_images = torch.stack(driving_images, dim=0)  # (T, C, H, W)
        driving_poses = torch.stack(driving_poses, dim=0)
        
        # For few-shot, stack along k dimension [k, C, H, W]
        few_shot_images = torch.stack(few_shot_images, dim=0)  # (K, C, H, W)
        few_shot_poses = torch.stack(few_shot_poses, dim=0)
        
        # Create output dictionary in format expected by fs_vid2vid trainer
        data = {
            'images': driving_images,  # (T, C, H, W) - target video frames
            'label': driving_poses,     # (T, C, H, W) - target pose sequence  
            'few_shot_images': few_shot_images,  # (K, C, H, W) - few-shot reference frames
            'few_shot_label': few_shot_poses,   # (K, C, H, W) - few-shot reference poses
        }
        
        return data