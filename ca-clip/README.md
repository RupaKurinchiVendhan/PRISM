# CA-CLIP: Compound-Aware CLIP for Image Restoration

A self-contained package for training and using CA-CLIP, an extension of CLIP that incorporates compositional degradation awareness through Jaccard-weighted contrastive learning.

## Installation

```bash
# Clone or extract this package
cd ca-clip-package

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

**Pre-degraded Images** (Required)
```
your_data/
├── clear/          # Ground truth clean images
├── blur/           # Single degradation
├── haze/
├── blur_haze/      # Compound degradations (underscore-separated)
└── ...
```

### 2. Configure Training

Edit `configs/train_predegraded.yml`:
```yaml
datasets:
  train:
    dataroot: /path/to/your_data
    clean_folder: clear
    num_variants: 64
    batch_size: 2
```

### 3. Train

```bash
# Single GPU
python -m ca_clip.train --config configs/train_predegraded.yml --gpus 0

# Multi-GPU
python -m ca_clip.train --config configs/train_predegraded.yml --gpus 0,1,2,3 --distributed
```

Or use the launcher script:
```bash
bash scripts/launch_training.sh --config configs/train_predegraded.yml --gpus 0,1,2,3
```

Importantly, the weighted contrastive loss is implemented in `ca_clip/open_clip/ca_clip_loss.py`.
```python
from ca_clip.open_clip.ca_clip_loss import CAClipLoss

loss_fn = CAClipLoss(temperature=0.1)
loss = loss_fn(
    clean_features=clean_feats,
    variant_features=variant_feats,
    degradation_labels=deg_labels
)
```

## Training Configurations
```yaml
ca_clip:
  num_variants: 256         # More variants
datasets:
  train:
    batch_size: 8            # Larger batch
    n_workers: 4             # Parallel loading
    patch_size: 256          # Larger patches
```

### Custom Loss Weights

Adjust loss balance:
```yaml
train:
  loss_type: l1              # l1, l2, or charbonnier
  weight: 1.0                # Reconstruction weight
  ca_clip_weight: 0.1        # CA-CLIP loss weight
```

### Resume Training

```yaml
path:
  resume_state: path/to/checkpoint.state
```

### Weights & Biases
```bash
# Logs are automatically uploaded to W&B when use_wandb: true
# View your dashboard at https://wandb.ai/your-username/ca-clip-universal-ir
```

You can customize the W&B project name in your config:
```yaml
wandb_project: my-custom-project-name
```

### Logs
Check training logs:
```bash
tail -f experiments/your_experiment_name/train_*.log
```

## Acknowledgments

This code is based on
- **DA-CLIP**: Degradation-Aware CLIP for image restoration
- **IR-SDE**: Image Restoration with Stochastic Differential Equations
- **OpenCLIP**: Open-source CLIP implementation
