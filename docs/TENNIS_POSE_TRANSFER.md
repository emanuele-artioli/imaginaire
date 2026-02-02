# Tennis Pose Transfer with fs_vid2vid

This guide explains how to train and run inference for pose transfer on tennis player videos using NVIDIA's fs_vid2vid (few-shot video-to-video synthesis) model.

## Overview

The fs_vid2vid model performs **pose-guided video synthesis** - given a reference image of a person and a sequence of target poses, it generates a video of that person performing the movements described by the poses.

**Architecture:**
- **Generator**: 91.1M parameters with spatially adaptive SPADE normalization
- **Discriminator**: 5.5M parameters (multi-scale)
- **Optical Flow**: FlowNet2 for temporal consistency
- **Perceptual Loss**: VGG19-based

## Prerequisites

### Environment Setup

```bash
# Create and activate conda environment
conda create -n imaginaire python=3.10 -y
conda activate imaginaire

# Install PyTorch with CUDA support
conda install pytorch=2.5.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install imaginaire dependencies
cd /home/itec/emanuele/imaginaire
pip install -r requirements.txt

# Build CUDA extensions (required for C++17 compatibility with PyTorch 2.5+)
cd imaginaire/third_party/correlation && python setup.py install
cd ../channelnorm && python setup.py install
cd ../resample2d && python setup.py install
cd ../bias_act && python setup.py install
cd ../upfirdn2d && python setup.py install
```

### Dataset Structure

The tennis dataset should be organized as follows:

```
/home/itec/emanuele/pointstream/experiments/dataset/
├── match_name/                    # e.g., djokovic_federer
│   ├── sequence_id/               # e.g., 012, 015, etc.
│   │   ├── crops/
│   │   │   └── person_id/         # e.g., id0, id1
│   │   │       ├── 00000.png      # RGB crop frames (512x512)
│   │   │       ├── 00001.png
│   │   │       └── ...
│   │   └── poses/
│   │       └── person_id/
│   │           ├── 00000.png      # Corresponding pose maps (512x512)
│   │           ├── 00001.png
│   │           └── ...
```

**Current dataset statistics:**
- 38 sequences across multiple tennis matches
- Maximum sequence length: 1324 frames
- Resolution: 512x512 RGB PNG files

## Configuration

The main configuration file is located at:
```
configs/projects/fs_vid2vid/tennis/ampO1.yaml
```

### Key Configuration Options

```yaml
# Training duration
max_epoch: 20
snapshot_save_iter: 100        # Save checkpoint every N iterations
snapshot_save_epoch: 1         # Also save at end of each epoch
image_save_iter: 50            # Save visualization every N iterations

# Trainer settings
trainer:
    type: imaginaire.trainers.fs_vid2vid
    reuse_gen_output: False    # IMPORTANT: Must be False for PyTorch 2.x
    amp_config:
        enabled: False         # IMPORTANT: Disable AMP to avoid graph issues

# Loss weights
    loss_weight:
        gan: 1.0
        feature_matching: 10.0
        perceptual: 10.0
        flow: 10.0

# Generator architecture
gen:
    type: imaginaire.generators.fs_vid2vid
    num_filters: 32
    max_num_filters: 1024
    num_downsamples: 5

# Dataset configuration
data:
    type: imaginaire.datasets.tennis_pose
    output_h_w: 512, 512
    train:
        roots:
            - /home/itec/emanuele/pointstream/experiments/dataset
        batch_size: 1
        initial_sequence_length: 4

# Inference settings
inference_args:
    driving_seq_index: 0       # Which sequence to use for driving poses
    few_shot_seq_index: 0      # Which sequence to use for reference appearance
    few_shot_frame_index: 0    # Starting frame for few-shot reference
```

## Training

### Running Training

```bash
cd /home/itec/emanuele/imaginaire

# Single GPU training
python train.py \
    --config configs/projects/fs_vid2vid/tennis/ampO1.yaml \
    --single_gpu

# Multi-GPU training (distributed)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --config configs/projects/fs_vid2vid/tennis/ampO1.yaml
```

### Training Output

Training creates a timestamped log directory:
```
logs/YYYY_MMDD_HHMM_SS_ampO1/
├── epoch_XXXXX_iteration_XXXXXXXXX_checkpoint.pt   # Model checkpoints
├── latest_checkpoint.txt                            # Points to latest checkpoint
├── images/                                          # Visualization samples
│   └── epoch_XXXXX_iteration_XXXXXXXXX.jpg
├── tensorboard/                                     # TensorBoard logs
└── wandb_id.txt                                     # W&B run ID (if enabled)
```

### Monitoring Training

```bash
# View TensorBoard logs
tensorboard --logdir logs/

# Check training progress
tail -f logs/*/training.log
```

### Expected Training Metrics

Typical loss values during early training:
- **GAN loss**: ~2.0-2.5
- **Feature Matching**: ~2.5-3.5
- **Perceptual**: ~2.5-3.5
- **Flow L1**: ~0.05-0.15
- **Flow Warp**: ~1.5-2.5
- **Flow Mask**: ~0.1-0.7

## Inference

### Running Inference

```bash
cd /home/itec/emanuele/imaginaire

python inference.py \
    --config configs/projects/fs_vid2vid/tennis/ampO1.yaml \
    --checkpoint logs/2026_0202_1136_46_ampO1/epoch_00002_iteration_000000100_checkpoint.pt \
    --output_dir inference_output \
    --single_gpu
```

### Inference on Different Sequences

To run inference on a different sequence, modify `inference_args` in the config:

```yaml
inference_args:
    driving_seq_index: 5       # Use sequence 5 for driving poses
    few_shot_seq_index: 5      # Use sequence 5 for reference appearance
    few_shot_frame_index: 0    # Start from frame 0
```

Or create a separate inference config file.

### Cross-Identity Transfer

To transfer the appearance of one person to the poses of another:

```yaml
inference_args:
    driving_seq_index: 10      # Poses from sequence 10
    few_shot_seq_index: 5      # Appearance from sequence 5
    few_shot_frame_index: 0
```

### Inference Output

```
inference_output/
├── 000/                       # Sequence output directory
│   ├── 00000.jpg              # Individual frame outputs
│   ├── 00001.jpg
│   └── ...
└── 000.mp4                    # Generated video (15 fps)
```

Each JPG contains a visualization grid showing:
- Input pose
- Generated image
- (Optional) Ground truth for comparison

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` to 1
   - Reduce `initial_sequence_length` to 2
   - Use gradient checkpointing

2. **"Trying to backward through the graph a second time"**
   - Ensure `reuse_gen_output: False` in config
   - Ensure `amp_config.enabled: False`

3. **FlowNet2 weights not loading**
   - This is expected - the Google Drive link is broken
   - FlowNet2 will initialize with random weights and learn during training

4. **PyTorch 2.x compatibility errors**
   - Ensure CUDA extensions are compiled with C++17 flag
   - Use the modified trainer code that handles `_step_count` deprecation

### Performance Tips

1. **For faster training:**
   - Enable `num_workers: 4` in data config
   - Use mixed precision if stable: `amp_config.enabled: True`

2. **For better quality:**
   - Train for more epochs (50-100)
   - Use larger `num_filters` (64 instead of 32)
   - Enable model averaging: `model_average_config.enabled: True`

## File Locations

| Component | Path |
|-----------|------|
| Config | `configs/projects/fs_vid2vid/tennis/ampO1.yaml` |
| Dataset loader | `imaginaire/datasets/tennis_pose.py` |
| Trainer | `imaginaire/trainers/fs_vid2vid.py` |
| Generator | `imaginaire/generators/fs_vid2vid.py` |
| Discriminator | `imaginaire/discriminators/fs_vid2vid.py` |
| Checkpoints | `logs/*/epoch_*_checkpoint.pt` |

## Example Commands

```bash
# Activate environment
conda activate imaginaire
cd /home/itec/emanuele/imaginaire

# Train for 20 epochs
python train.py --config configs/projects/fs_vid2vid/tennis/ampO1.yaml --single_gpu

# Run inference with latest checkpoint
CKPT=$(cat logs/2026_0202_1136_46_ampO1/latest_checkpoint.txt)
python inference.py \
    --config configs/projects/fs_vid2vid/tennis/ampO1.yaml \
    --checkpoint logs/2026_0202_1136_46_ampO1/$CKPT \
    --output_dir inference_output \
    --single_gpu

# List all available sequences
python -c "
from imaginaire.datasets.tennis_pose import Dataset
from imaginaire.config import Config
cfg = Config('configs/projects/fs_vid2vid/tennis/ampO1.yaml')
ds = Dataset(cfg)
for i, seq in enumerate(ds.sequences):
    print(f'{i}: {seq[\"match\"]}/{seq[\"sequence\"]}/{seq[\"person\"]} ({seq[\"num_frames\"]} frames)')
"
```

## References

- [NVIDIA Imaginaire](https://github.com/NVlabs/imaginaire)
- [Few-shot vid2vid Paper](https://arxiv.org/abs/1910.12713)
- [Original vid2vid Paper](https://arxiv.org/abs/1808.06601)
