#!/usr/bin/env python3
"""
Convert the tennis dataset to LMDB format for fs_vid2vid training.
Your dataset structure:
  dataset/
    djokovic_federer/
      012/
        crops/
          id0/
            00000.png, 00001.png, ...
          id1/
            00000.png, 00001.png, ...
        poses/
          id0/
            00000.png, 00001.png, ...
          id1/
            00000.png, 00001.png, ...
    ...
"""

import os
import sys
import cv2
import lmdb
import json
from pathlib import Path
from tqdm import tqdm

def convert_tennis_to_lmdb(input_root, output_root):
    """Convert tennis dataset to LMDB format"""
    
    # Create output directories
    os.makedirs(output_root, exist_ok=True)
    
    # Open LMDB database
    db_path = os.path.join(output_root, 'tennis_data.lmdb')
    env = lmdb.open(db_path, map_size=int(1e12))
    
    key_idx = 0
    
    # Iterate through each video sequence
    for match_dir in sorted(os.listdir(input_root)):
        match_path = os.path.join(input_root, match_dir)
        if not os.path.isdir(match_path):
            continue
            
        print(f"\nProcessing match: {match_dir}")
        
        # Each sequence ID (012, 015, etc.) is a different shot
        for seq_dir in sorted(os.listdir(match_path)):
            seq_path = os.path.join(match_path, seq_dir)
            if not os.path.isdir(seq_path):
                continue
            
            crops_path = os.path.join(seq_path, 'crops')
            poses_path = os.path.join(seq_path, 'poses')
            
            if not os.path.isdir(crops_path) or not os.path.isdir(poses_path):
                continue
            
            # Each person ID (id0, id1, etc.)
            for person_id in sorted(os.listdir(crops_path)):
                person_crops_path = os.path.join(crops_path, person_id)
                person_poses_path = os.path.join(poses_path, person_id)
                
                if not os.path.isdir(person_crops_path) or not os.path.isdir(person_poses_path):
                    continue
                
                # Get all frames for this person
                crop_frames = sorted([f for f in os.listdir(person_crops_path) if f.endswith('.png')])
                pose_frames = sorted([f for f in os.listdir(person_poses_path) if f.endswith('.png')])
                
                # Ensure we have the same number of frames
                num_frames = min(len(crop_frames), len(pose_frames))
                
                if num_frames < 2:
                    print(f"  Skipping {match_dir}/{seq_dir}/{person_id} - too few frames")
                    continue
                
                print(f"  Adding {match_dir}/{seq_dir}/{person_id} ({num_frames} frames)")
                
                # Store metadata
                metadata_key = f'metadata_{key_idx}'.encode()
                metadata = {
                    'num_frames': num_frames,
                    'match': match_dir,
                    'sequence': seq_dir,
                    'person': person_id,
                    'height': 512,
                    'width': 512
                }
                
                with env.begin(write=True) as txn:
                    txn.put(metadata_key, json.dumps(metadata).encode())
                
                # Store frames
                for frame_idx in range(num_frames):
                    # Read crop image
                    crop_path = os.path.join(person_crops_path, crop_frames[frame_idx])
                    crop_img = cv2.imread(crop_path)
                    
                    # Read pose image
                    pose_path = os.path.join(person_poses_path, pose_frames[frame_idx])
                    pose_img = cv2.imread(pose_path)
                    
                    # Store as binary data
                    crop_key = f'crop_{key_idx}_{frame_idx}'.encode()
                    pose_key = f'pose_{key_idx}_{frame_idx}'.encode()
                    
                    with env.begin(write=True) as txn:
                        txn.put(crop_key, cv2.imencode('.png', crop_img)[1].tobytes())
                        txn.put(pose_key, cv2.imencode('.png', pose_img)[1].tobytes())
                
                key_idx += 1
    
    # Write summary
    summary = {'total_sequences': key_idx}
    with env.begin(write=True) as txn:
        txn.put(b'summary', json.dumps(summary).encode())
    
    env.close()
    print(f"\nDataset conversion complete. LMDB database saved to: {db_path}")
    print(f"Total sequences: {key_idx}")

if __name__ == '__main__':
    input_root = '/home/itec/emanuele/pointstream/experiments/dataset'
    output_root = '/home/itec/emanuele/imaginaire/datasets/tennis_dataset/lmdb'
    
    if not os.path.exists(input_root):
        print(f"Error: Input root {input_root} does not exist")
        sys.exit(1)
    
    convert_tennis_to_lmdb(input_root, output_root)
