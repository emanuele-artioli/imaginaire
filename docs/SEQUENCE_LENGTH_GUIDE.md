# Understanding Sequence Length and Few-Shot Parameters

When training fs_vid2vid on the tennis dataset, you'll see messages like:

```
Updated sequence length to 4, few_shot_K to 1
Available sequences: 38 (was 38)
```

This guide explains what these values mean and how to tweak them.

## Parameter Definitions

### Sequence Length

**What it is:** The number of consecutive video frames processed together during training.

- **Value:** Integer (typically 1, 2, 4, 8, 16...)
- **Example:** `sequence_length = 4` means the model processes 4 consecutive frames at a time
- **Default:** Starts at 1, increases over training epochs

**Visual representation:**

```
Sequence with 4 frames:
Frame 0 → Frame 1 → Frame 2 → Frame 3
[Single training example]
```

### Few-Shot K (few_shot_K)

**What it is:** The number of reference frames used to guide pose transfer.

- **Value:** Integer (typically 1-3)
- **Example:** `few_shot_K = 1` means 1 reference frame is used
- **Default:** 1 (one reference frame per sequence)

**Visual representation:**

```
Training data structure:
├─ Few-shot frames (K=1): [Reference Frame 0]
│  └─ Used to capture appearance/identity
│
└─ Driving frames (sequence_length=4): [Frame 1, Frame 2, Frame 3, Frame 4]
   └─ Poses to follow (target output)
```

### Available Sequences

**What it is:** The number of video sequences that meet the minimum length requirement.

- **Value:** Integer (0-38 in your dataset)
- **Example:** "Available sequences: 38 (was 38)" means all 38 sequences are usable
- **Requirement:** Minimum length = `sequence_length + few_shot_K`

**Why it matters:**
```
Minimum frames needed = sequence_length + few_shot_K + 1 extra
(for sampling randomness during training)

Example with sequence_length=4, few_shot_K=1:
Minimum = 4 + 1 = 5 frames required per sequence
All 38 sequences have > 1324 frames, so all are available
```

## How These Change During Training

The model uses a **progressive training strategy**:

```
Epoch 0:          sequence_length = 1 (single frame, no temporal info)
Epoch 1-4:        sequence_length = 4 (after init_temporal_network)
Epoch 5-9:        sequence_length = 8 (doubled after 5 epochs)
Epoch 10-14:      sequence_length = 16 (doubled again)
...
```

This is controlled by two config parameters:

```yaml
# Config parameters
single_frame_epoch: 0           # When to add temporal network
num_epochs_temporal_step: 5     # How many epochs before doubling
data:
    train:
        initial_sequence_length: 4     # Starting length after temporal
        max_sequence_length: 8         # Never exceed this length
```

## Configuration Reference

### In `ampO1.yaml`:

```yaml
# Temporal training schedule
single_frame_epoch: 0              # Start temporal training immediately
num_epochs_temporal_step: 5        # Double sequence every 5 epochs

data:
    train:
        initial_sequence_length: 4    # Initial frames to process
        max_sequence_length: 8        # Cap at 8 frames

    initial_few_shot_K: 1             # Number of reference frames
```

## How to Tweak These Values

### 1. Increasing Sequence Length

**Effect:** Process longer video sequences, better temporal consistency

**Change:**
```yaml
data:
    train:
        initial_sequence_length: 8    # Instead of 4
        max_sequence_length: 16       # Instead of 8
```

**Tradeoffs:**
- ✅ Better temporal consistency and smoothness
- ✅ Model learns longer-term dependencies
- ❌ Higher GPU memory usage (4x for 8 frames vs 4 frames)
- ❌ Slower training (more frames per batch)
- ❌ May lose some sequences if they're too short

**Expected impact on "Available sequences":**
```
sequence_length=4, few_shot_K=1: Need 5+ frames → 38 sequences
sequence_length=8, few_shot_K=1: Need 9+ frames → 38 sequences (yours are long)
sequence_length=16, few_shot_K=1: Need 17+ frames → 38 sequences
```

### 2. Increasing Few-Shot K

**Effect:** More reference frames for appearance consistency

**Change:**
```yaml
data:
    initial_few_shot_K: 2    # Instead of 1
```

**Tradeoffs:**
- ✅ Richer appearance information
- ✅ More stable identity preservation
- ❌ Slight GPU memory increase
- ❌ Fewer driving frames available (N-K frames per sequence)
- ❌ May lose sequences if you increase it too much

**Expected impact:**
```
sequence_length=4, few_shot_K=1: Need 5 frames, 38 available
sequence_length=4, few_shot_K=2: Need 6 frames, 38 available
sequence_length=4, few_shot_K=3: Need 7 frames, 38 available (still fine)
```

### 3. Controlling Temporal Progression

**Effect:** How quickly sequence length increases during training

**Option A: Faster progression**
```yaml
num_epochs_temporal_step: 2    # Double every 2 epochs instead of 5
```

**Option B: Slower progression**
```yaml
num_epochs_temporal_step: 10   # Double every 10 epochs instead of 5
```

**Option C: No progression**
```yaml
num_epochs_temporal_step: 1000  # Never double (cap hits max instead)
single_frame_epoch: 0
data:
    train:
        initial_sequence_length: 8
        max_sequence_length: 8  # Stay at 8
```

## Typical Training Evolution

With default config (`initial_sequence_length=4`, `num_epochs_temporal_step=5`):

```
Epoch 0:
  Output: "------- Updating sequence length to 1 -------"
  Processing: 1 frame per batch
  GPU memory: Minimal
  Training: Stable but limited temporal info

Epoch 1:
  Output: "------ Now start training 4 frames -------"
  Processing: 4 frames per batch
  GPU memory: 4x baseline
  Training: Adding temporal awareness

Epoch 5:
  Output: "------- Updating sequence length to 8 -------"
  Processing: 8 frames per batch
  GPU memory: 8x baseline
  Training: Better temporal consistency

Epoch 10:
  Output: "------- Updating sequence length to 16 -------"
  Processing: 16 frames per batch
  GPU memory: 16x baseline (may exceed VRAM!)
  Training: Rich temporal context

Epoch 20+:
  Still processing 16 frames (hits max_sequence_length limit)
```

## Recommended Configurations

### For Limited GPU Memory (8GB):

```yaml
single_frame_epoch: 0
num_epochs_temporal_step: 10    # Slower progression
data:
    train:
        initial_sequence_length: 2
        max_sequence_length: 4    # Keep sequences short
    initial_few_shot_K: 1
```

### For Balanced Training (16GB):

```yaml
single_frame_epoch: 0
num_epochs_temporal_step: 5     # Default
data:
    train:
        initial_sequence_length: 4
        max_sequence_length: 8
    initial_few_shot_K: 1
```

### For Better Quality (32GB+):

```yaml
single_frame_epoch: 0
num_epochs_temporal_step: 3     # Faster progression
data:
    train:
        initial_sequence_length: 8
        max_sequence_length: 32   # Long sequences
    initial_few_shot_K: 2        # More reference frames
```

## Monitoring the Impact

When you adjust these parameters:

1. **Check "Available sequences" count:**
   ```
   print(f'Available sequences: {len(self.sequences)}')
   ```
   If this drops significantly, your sequences are too short

2. **Monitor GPU memory:**
   ```bash
   nvidia-smi
   ```
   If it exceeds your VRAM, reduce `max_sequence_length`

3. **Check training loss:**
   - Initially increases (more frames = harder task)
   - Should stabilize and decrease over epochs

4. **Evaluate visual quality:**
   - More frames = smoother, less jittery motion
   - More reference frames = better identity consistency

## Quick Reference Table

| Config | Frames | Memory | Quality | Speed | Stability |
|--------|--------|--------|---------|-------|-----------|
| seq=2 | Low | 1x | Poor | Fast | Very High |
| seq=4 | Medium | 2x | Good | Medium | High |
| seq=8 | High | 4x | Better | Slow | Medium |
| seq=16 | Very High | 8x | Best | Very Slow | Lower |
| k=1 | - | Baseline | OK | Baseline | High |
| k=2 | - | 1.1x | Better | -5% | Medium |
| k=3 | - | 1.2x | Best | -10% | Lower |

## Troubleshooting

**Problem: "Available sequences: 0 (was 38)"**
- Your new `sequence_length + few_shot_K` exceeds all sequence lengths
- Solution: Reduce sequence length or few_shot_K

**Problem: GPU out of memory errors**
- Sequence length too high for your GPU
- Solution: Reduce `max_sequence_length` or disable progressive training

**Problem: Training is very slow after epoch 5**
- Sequence length has grown too large
- Solution: Lower `max_sequence_length` or increase `num_epochs_temporal_step`

**Problem: Video output is jittery**
- Sequence length too short for temporal consistency
- Solution: Increase `initial_sequence_length`
