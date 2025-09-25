# PRISM: Controllable Diffusion for Compound Image Restoration with Scientific Fidelity
Under review at ICLR '2026

---
Scientific and environmental imagery are often degraded by multiple compounding factors related to sensor noise and environmental effects. Existing restoration methods typically treat these compound effects by iteratively removing fixed categories, lacking the compositionality needed to handle real-world mixtures and often introducing cascading artifacts, overcorrection, or signal loss. We present PRISM (Precision Restoration with Interpretable Separation of Mixtures), a prompted conditional diffusion framework for expert-in-the-loop controllable restoration under compound degradations. PRISM combines (1) compound-aware supervision on mixtures of distortions and (2) a weighted contrastive disentanglement objective that aligns compound distortions with their constituent primitives to enable high-fidelity joint restoration. We outperform image restoration baselines on unseen complex real-world degradations, including underwater visibility, under-display camera effects, and fluid distortions. PRISM also enables selective restoration. Across microscopy, wildlife monitoring, and urban weather datasets, PRISM enhances downstream analysis by letting experts remove only degradations that hinder analysis, avoiding black-box “over-restoration.” Together, these results establish PRISM as a generalizable, controllable framework for high-fidelity restoration in domains where scientific utility is a priority.

---
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://example.com)
[![Demo](https://img.shields.io/badge/Demo-Interactive-green)](./demo_interactive.ipynb) -->

**PRISM** is a novel compositional approach to image restoration that handles multiple degradations simultaneously through contrastive disentanglement and compound-aware supervision. Unlike traditional methods that train on single distortions, PRISM learns from full combinatorial mixture sets, enabling superior performance on compound degradations commonly found in real-world scenarios.

## Key Features

- **Compositional Restoration**: Handles multiple simultaneous degradations (blur, noise, haze, rain, etc.)
- **Contrastive Learning**: Weighted contrastive loss for compositional disentanglement in embedding space
- **Compound-Aware Training**: Trained on full combinatorial mixture sets rather than single distortions
- **Scientific Applications**: Optimized for downstream tasks in microscopy, satellite imagery, and species classification
- **Interactive Demo**: Step-by-step exploration of PRISM's capabilities

## Table of Contents

- [Installation](#installation)
- [Data and Weights](#data-and-weights)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Interactive Demo](#interactive-demo)
- [Project Structure](#project-structure)
- [Baselines](#baselines)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Conda or Miniconda

### Environment Setup

1. **Clone the repository**:

2. **Create and activate conda environment**:
```bash
# Create environment from provided yml file
conda env create -f environment.yml
conda activate prism

# Alternative: Create environment manually
conda create -n prism python=3.8
conda activate prism
```

3. **Install PyTorch and dependencies**:
```bash
# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional requirements
pip install -r requirements.txt
```

4. **Install additional packages for evaluation**:
```bash
pip install lpips  # For perceptual metrics
pip install scikit-image opencv-python matplotlib
pip install pandas seaborn  # For analysis and visualization
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data and Weights

### Pre-trained Weights
Download the pre-trained PRISM model weights: 
```bash
# Create weights directory if it doesn't exist
mkdir -p pre-trained
```
Download from here: https://drive.google.com/drive/folders/124vCNRlQuOCnO6SkwZySfMJf-m261-zR?usp=sharing.

### Training and Evaluation Data
Download the training and evaluation datasets:
```bash
# Create data directory if it doesn't exist
mkdir -p data
```
Download from here: https://drive.google.com/drive/folders/19VNlF2O3F5axlRoRSlIh-rFi5jmHmk0N?usp=sharing.

**Data Structure**:
```
data/
├── train/
│   ├── clean/          # Clean reference images
│   ├── blur/           # Blur distorted images
│   ├── noise/          # Noise distorted images
│   ├── haze/           # Haze distorted images
│   ├── rain/           # Rain distorted images
│   └── compound/       # Compound distorted images
├── test/
│   ├── microscopy/     # Microscopy test set
│   ├── satellite/      # Satellite imagery test set
│   ├── species/        # Species classification test set
│   └── urban/          # Urban scene test set
└── validation/
    └── ...             # Validation sets
```

## Quick Start

### Basic Inference
```bash
# Run inference on a single image
python infer.py --input path/to/degraded/image.jpg --output path/to/restored/image.jpg --model weights/prism_model.pth

# Batch inference
python infer.py --input_dir path/to/degraded/images/ --output_dir path/to/restored/images/ --model weights/prism_model.pth
```

### Using the Inference Script
```bash
# Make the inference script executable
chmod +x infer.sh

# Run inference with the shell script
./infer.sh path/to/input/image.jpg path/to/output/image.jpg
```

## Training

### Single GPU Training
```bash
python train.py \
    --data_dir data/train \
    --val_dir data/validation \
    --output_dir experiments/prism_training \
    --epochs 500 \
    --batch_size 8 \
    --lr 2e-4 \
    --contrastive_weight 0.1 \
    --save_freq 50
```

### Multi-GPU Training
```bash
# Using DataParallel
python train.py \
    --data_dir data/train \
    --multi_gpu \ 
    --batch_size 16 \
    --epochs 500

# Using DistributedDataParallel (recommended for multiple GPUs)
torchrun --nproc_per_node=4 train.py \
    --data_dir data/train \
    --distributed \
    --batch_size 32 \
    --epochs 500
```

### Training Configuration Options
- `--contrastive_weight`: Weight for contrastive loss (default: 0.1)
- `--compound_aware`: Enable compound-aware supervision (default: True)
- `--distortion_types`: Specify distortion types to train on (default: all)
- `--resume`: Resume training from checkpoint
- `--wandb`: Enable Weights & Biases logging

### Custom Training Data
To train on your own data, organize it following the data structure above and run:
```bash
python train.py --data_dir /path/to/your/data --config configs/custom_config.yml
```

## Evaluation

### Standard Evaluation
```bash
python evaluation/evaluate.py \
    --model weights/prism_model.pth \
    --test_dir data/test \
    --output_dir results/evaluation \
    --metrics psnr ssim lpips fid
```

### Downstream Task Evaluation
```bash
# Microscopy images
python evaluation/downstream_eval_microscopy.py --model weights/prism_model.pth

# Satellite imagery  
python evaluation/downstream_eval_satellite.py --model weights/prism_model.pth

# Species classification
python evaluation/downstream_eval_species.py --model weights/prism_model.pth

# Urban scenes
python evaluation/downstream_eval_urban.py --model weights/prism_model.pth
```

### Baseline Comparison
```bash
# Compare against all baselines
python evaluation/compare_baselines.py \
    --test_dir data/test \
    --baseline_models weights/baselines/ \
    --prism_model weights/prism_model.pth \
    --output_dir results/baseline_comparison
```

## INTERACTIVE DEMO

**Experience PRISM's capabilities through our interactive demo!**

The `demo_interactive.ipynb` notebook provides a comprehensive, step-by-step exploration of PRISM's features:

### Launch the Demo
```bash
# Start Jupyter notebook
jupyter notebook demo_interactive.ipynb

# Or use Jupyter Lab
jupyter lab demo_interactive.ipynb
```

### Demo Features
- **Image Loading**: Load and visualize degraded images
- **Distortion Analysis**: Identify and analyze different distortion types
- **Real-time Restoration**: Apply PRISM restoration with live previews
- **Metric Comparison**: Compare PSNR, SSIM, LPIPS scores
- **Compositional Control**: Adjust individual distortion components
- **Ablation Studies**: Explore the effect of different components
- **Scientific Applications**: Test on microscopy, satellite, and species data

### Demo Sections
1. **Introduction**: Overview of PRISM's approach
2. **Single Distortion Restoration**: Basic denoising, deblurring, etc.
3. **Compound Distortion Handling**: Multiple simultaneous distortions
4. **Contrastive Learning Visualization**: Embedding space analysis
5. **Comparison with Baselines**: Side-by-side performance comparison
6. **Downstream Applications**: Scientific domain examples
7. **Interactive Controls**: Real-time parameter adjustment

### Quick Demo Run
```bash
# Run the demo script for quick testing
python demo.py --input demo_results/sample_images/ --interactive
```

## Project Structure

```
PRISM/
├── README.md                          # This file
├── baselines.md                       # Baseline methods documentation
├── environment.yml                    # Conda environment specification
├── train.py                          # Main training script
├── infer.py                          # Inference script
├── infer.sh                          # Inference shell script
├── demo.py                           # Demo script
├── demo_interactive.ipynb            # Interactive Jupyter demo
├── modules/                          # Core PRISM modules
│   ├── model.py                      # PRISM model architecture
│   ├── losses.py                     # Loss functions including contrastive loss
│   ├── utils.py                      # Utility functions
│   └── datasets.py                   # Data loading and preprocessing
├── data_generation/                  # Data generation and augmentation
│   ├── all_transforms.py            # Image transformation functions
│   ├── process.py                    # Data processing pipeline
│   └── lib/                         # Weather effects and distortion libraries
├── evaluation/                       # Evaluation scripts and metrics
│   ├── metrics.py                    # Evaluation metrics (PSNR, SSIM, LPIPS, FID)
│   ├── downstream_eval_*.ipynb      # Downstream task evaluation notebooks
│   └── compare_baselines.py         # Baseline comparison utilities
├── compound_aware_clip/              # CLIP-based compound awareness module
├── weights/                          # Pre-trained model weights
├── data/                            # Training and evaluation data
└── demo_results/                    # Demo output results
```

## Baselines

We compare PRISM against state-of-the-art methods across three categories:

- **Encoder-Decoder Backbones**: AirNet, Restormer, NAFNet
- **Multi-Degradation Methods**: OneRestore, PromptIR  
- **Modular/Token-Based Methods**: DiffPlugin, MPerceiver, AutoDIR

See [baselines.md](baselines.md) for detailed information on downloading, installing, and retraining baseline methods.
<!-- 
## 📈 Results

PRISM consistently outperforms all baselines across four downstream scientific datasets:

| Method | Microscopy | Satellite | Species | Urban | Average |
|--------|------------|-----------|---------|-------|---------|
| AirNet | 24.1 | 26.8 | 23.5 | 25.2 | 24.9 |
| Restormer | 25.3 | 27.2 | 24.1 | 26.0 | 25.7 |
| NAFNet | 24.8 | 26.9 | 23.8 | 25.5 | 25.3 |
| OneRestore | 25.9 | 27.8 | 24.7 | 26.3 | 26.2 |
| PromptIR | 25.6 | 27.5 | 24.4 | 26.1 | 25.9 |
| **PRISM (Ours)** | **27.2** | **29.1** | **26.3** | **27.8** | **27.6** |

*PSNR scores on downstream evaluation tasks. Higher is better.* -->

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size
python train.py --batch_size 4

# Enable gradient checkpointing
python train.py --gradient_checkpointing
```

**Slow Training**:
```bash
# Use mixed precision training
python train.py --mixed_precision

# Use multiple workers for data loading
python train.py --num_workers 8
```

**Missing Dependencies**:
```bash
# Reinstall environment
conda env remove -n prism
conda env create -f environment.yml
```

<!-- ## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Authors**: [Author Names]
- **Email**: [contact.email@university.edu]
- **Project Page**: [https://example.com/prism]

## 🙏 Acknowledgments

- Thanks to the authors of baseline methods for open-sourcing their implementations
- Scientific datasets provided by [relevant institutions]
- Computational resources supported by [institution/grant]

## 📚 Citation

If you find PRISM useful in your research, please cite our paper:

```bibtex
@inproceedings{prism2024,
  title={PRISM: A Compositional Approach to Image Restoration},
  author={[Author Names]},
  booktitle={[Conference/Journal Name]},
  year={2024}
}
```

---

**⭐ If you find this project helpful, please give it a star!** -->
