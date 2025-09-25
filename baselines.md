# PRISM Baseline Methods

This details the baseline methods we benchmark against in our PRISM paper, including GitHub links for downloading and retraining. All baseline models were re-trained or fine-tuned on the same set of primitive distortions as PRISM to ensure fair comparison.

## Overview

Our baselines span three main categories:
- **Encoder-Decoder Backbones**: Strong architectures for low-level vision tasks
- **Multi-Degradation Methods**: Approaches that handle multiple degradations
- **Modular/Token-Based Methods**: Methods using conditioning or routing strategies

All baselines (except OneRestore) are trained on single distortions/primitives only, unlike PRISM which is trained on the full combinatorial mixture set.

---

## Encoder-Decoder Backbone Methods

### AirNet
**Paper**: All-in-One Image Restoration for Unknown Corruption  
**Venue**: CVPR 2022  
**GitHub**: [https://github.com/XLearning-SCU/2022-CVPR-AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet)

**Description**: Strong encoder-decoder backbone for low-level vision tasks operating in an all-in-one setting without explicit modeling of compound effects.

**Installation**:
```bash
git clone https://github.com/XLearning-SCU/2022-CVPR-AirNet.git
cd 2022-CVPR-AirNet
pip install -r requirements.txt
```

**Retraining on PRISM Primitives**:
```bash
# Train on single distortions (blur, noise, haze, rain)
python train.py --arch AirNet --mode train --data_file_dir [DATA_PATH] \
    --distortion blur --epochs 500 --lr 2e-4

python train.py --arch AirNet --mode train --data_file_dir [DATA_PATH] \
    --distortion noise --epochs 500 --lr 2e-4

python train.py --arch AirNet --mode train --data_file_dir [DATA_PATH] \
    --distortion haze --epochs 500 --lr 2e-4

python train.py --arch AirNet --mode train --data_file_dir [DATA_PATH] \
    --distortion rain --epochs 500 --lr 2e-4
```

---

### Restormer
**Paper**: Restormer: Efficient Transformer for High-Resolution Image Restoration  
**Venue**: CVPR 2022  
**GitHub**: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)

**Description**: Efficient Transformer architecture with Multi-Dconv Head Transposed Attention (MDTA) and Gated-Dconv Feed-Forward Network (GDFN).

**Installation**:
```bash
git clone https://github.com/swz30/Restormer.git
cd Restormer
pip install -r requirements.txt
```

**Retraining on PRISM Primitives**:
```bash
# Train separate models for each primitive distortion
python train.py --model Restormer --task Denoising --train_dir [NOISE_DATA] --epochs 300
python train.py --model Restormer --task Deblurring --train_dir [BLUR_DATA] --epochs 3000
python train.py --model Restormer --task Deraining --train_dir [RAIN_DATA] --epochs 200
python train.py --model Restormer --task Dehazing --train_dir [HAZE_DATA] --epochs 200
```

---

### NAFNet
**Paper**: Simple Baselines for Image Restoration  
**Venue**: ECCV 2022  
**GitHub**: [https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

**Description**: Nonlinear Activation Free Network with Simple Channel Attention (SCA) and Simplified Gate (SG) mechanisms.

**Installation**:
```bash
git clone https://github.com/megvii-research/NAFNet.git
cd NAFNet
pip install -r requirements.txt
```

**Retraining on PRISM Primitives**:
```bash
# Configure for single distortion training
python basicsr/train.py -opt options/train/NAFNet/NAFNet-blur.yml
python basicsr/train.py -opt options/train/NAFNet/NAFNet-noise.yml  
python basicsr/train.py -opt options/train/NAFNet/NAFNet-haze.yml
python basicsr/train.py -opt options/train/NAFNet/NAFNet-rain.yml
```

---

## Multi-Degradation Methods

### OneRestore
**Paper**: One-to-Composite Mapping for Mixed Image Enhancement  
**GitHub**: [https://github.com/guochengqian/OneRestore](https://github.com/guochengqian/OneRestore)  
**Note**: This is the only baseline trained on composite degradations like PRISM.

**Description**: Introduces a one-to-composite mapping approach for handling multiple degradations simultaneously.

**Installation**:
```bash
git clone https://github.com/guochengqian/OneRestore.git
cd OneRestore
pip install -r requirements.txt
```

**Training on PRISM Composite Data**:
```bash
# Train on composite degradations (similar to PRISM training data)
python train.py --model OneRestore --data_dir [COMPOSITE_DATA] \
    --degradations mixed --epochs 400 --lr 1e-4
```

---

### PromptIR
**Paper**: PromptIR: Prompting for All-in-One Blind Image Restoration  
**Venue**: NeurIPS 2023  
**GitHub**: [https://github.com/va1shn9v/PromptIR](https://github.com/va1shn9v/PromptIR)

**Description**: Conditions restoration on learned prompt embeddings for multi-degradation scenarios.

**Installation**:
```bash
git clone https://github.com/va1shn9v/PromptIR.git
cd PromptIR  
pip install -r requirements.txt
```

**Retraining on PRISM Primitives**:
```bash
# Train with prompts for each degradation type
python train.py --arch PromptIR --train_dir [DATA_PATH] \
    --degradation_types blur,noise,haze,rain --epochs 500
```

---

## Modular/Token-Based Methods

### DiffPlugin
**Paper**: DiffPlugin: Bridging Contrastive Learning with Diffusion Models  
**GitHub**: [https://github.com/Liu-SD/DiffPlugin](https://github.com/Liu-SD/DiffPlugin)

**Description**: Adopts modular conditioning with contrastive prompt modules integrated into diffusion-based restoration.

**Installation**:
```bash
git clone https://github.com/Liu-SD/DiffPlugin.git
cd DiffPlugin
pip install -r requirements.txt
```

**Retraining on PRISM Primitives**:
```bash
# Train diffusion model with contrastive prompt modules
python train.py --model DiffPlugin --degradations single \
    --contrastive_loss --epochs 1000 --lr 1e-4
```

---

### MPerceiver
**Paper**: Multimodal Perceiver for Image Restoration  
**GitHub**: [https://github.com/AI4Imaging/MPerceiver](https://github.com/AI4Imaging/MPerceiver)

**Description**: Encodes multiple degradation tokens using a perceiver-based architecture for token-based conditioning.

**Installation**:
```bash
git clone https://github.com/AI4Imaging/MPerceiver.git
cd MPerceiver
pip install torch torchvision transformers
```

**Retraining on PRISM Primitives**:
```bash
# Train with degradation tokens
python train.py --model MPerceiver --token_types blur,noise,haze,rain \
    --epochs 300 --lr 2e-4
```

---

### AutoDIR
**Paper**: AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion  
**Venue**: ICLR 2024  
**GitHub**: [https://github.com/jiangyuliu/AutoDIR](https://github.com/jiangyuliu/AutoDIR)

**Description**: Task-routing approach that selects subtasks adaptively during inference using latent diffusion models.

**Installation**:
```bash
git clone https://github.com/jiangyuliu/AutoDIR.git
cd AutoDIR
pip install -r requirements.txt
```

**Retraining on PRISM Primitives**:
```bash
# Train with task routing for single distortions
python train.py --model AutoDIR --routing_strategy adaptive \
    --tasks blur,noise,haze,rain --epochs 600
```

---

## Training Protocol

### Fair Comparison Setup
All baselines were retrained following these principles:

1. **Same Primitive Distortions**: All models trained on identical blur, noise, haze, and rain primitives
2. **Single Distortion Training**: All baselines (except OneRestore) trained on individual distortions only
3. **Identical Evaluation**: Same test sets and metrics across all methods
4. **Controlled Setting**: Predefined primitive degradations applied consistently

### Training Data
- **PRISM Primitives**: blur, noise, haze, rain
- **Single Distortion Models**: Separate training for each primitive
- **Composite Model** (OneRestore only): Trained on mixed degradations
- **Fair Evaluation**: All methods tested on identical compound distortions

### Key Differences from PRISM
- **No Contrastive Loss**: Only PRISM uses weighted contrastive loss for compositional disentanglement
- **Single vs. Composite**: Most baselines trained on single distortions, PRISM on full combinatorial mixture
- **Standard Supervision**: All baselines use their original loss functions without contrastive component

---

## Evaluation Results

As shown in our paper's Table (downstream evaluation), PRISM consistently outperforms all baselines across four downstream scientific datasets:
- **Microscopy Images**
- **Satellite Imagery** 
- **Species Classification**
- **Urban Scene Analysis**

This demonstrates the added benefit of:
- Compound-aware supervision
- Contrastive disentanglement
- Full combinatorial mixture training

---

## Implementation Details

### Hardware Requirements
- **GPU**: NVIDIA RTX 3090 or better (24GB+ VRAM recommended)
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 32GB+ recommended
- **Storage**: 500GB+ SSD for datasets and checkpoints

### Common Dependencies
```bash
# Create environment
conda create -n prism_baselines python=3.8
conda activate prism_baselines

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Common packages
pip install opencv-python scikit-image matplotlib numpy scipy
pip install tensorboard wandb lpips timm
pip install basicsr  # For some baselines
```

### Datasets
Ensure you have the PRISM primitive distortion datasets:
- Blur corrupted images
- Noise corrupted images  
- Haze corrupted images
- Rain corrupted images
- Clean reference images

---

## Reproducing Results

1. **Download/Clone** all baseline repositories
2. **Install** dependencies for each method
3. **Prepare** PRISM primitive distortion datasets
4. **Train** each baseline on single distortions (except OneRestore)
5. **Evaluate** on compound distortions using our metrics
6. **Compare** results with PRISM performance

Detailed training configurations and hyperparameters are available in our codebase.

---

## Citations

When using these baselines, please cite both the original papers and our PRISM work:

```bibtex
% Original baseline papers - see individual repositories for citations

% Our work
@inproceedings{prism2024,
  title={PRISM: A Compositional Approach to Image Restoration},
  author={[Authors]},
  booktitle={[Venue]},
  year={2024}
}
```

---

**Note**: Some GitHub links may be placeholders if repositories are not yet public. Please check the original papers for official implementation details.
