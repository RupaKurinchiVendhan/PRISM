# PRISM: Controllable Diffusion for Compound Image Restoration with Scientific Fidelity
### ICLR 2026

---
Scientific and environmental imagery are often degraded by multiple compounding factors related to sensor noise and the environment. Existing restoration methods typically treat these mixed effects by iteratively removing fixed categories, but they assume degradations occur in isolation and therefore cannot flexibly model real-world mixtures, often introducing cascading artifacts, overcorrection, or signal loss. Moreover, current supervised approaches rely on paired ground-truth data, which may be unavailable or impossible to simulate in many domains. We present \textbf{PRISM} (\textbf{P}recision \textbf{R}estoration with \textbf{I}nterpretable \textbf{S}eparation of \textbf{M}ixtures), a prompted conditional diffusion framework for {\emph{expert-guided restoration}} under compound degradations. PRISM combines (1) compound-aware supervision on mixtures of distortions and (2) a weighted contrastive disentanglement objective that aligns compound distortions with their constituent primitives to enable high-fidelity joint restoration. Our compound-aware latent space enables both automated restoration and generalization to unseen combinations of degradations. We outperform image restoration baselines on unseen complex real-world degradations, including underwater visibility, under-display camera effects, and fluid distortions. PRISM also enables selective restoration. Across microscopy, wildlife monitoring, and urban weather datasets, our method allows experts to remove only degradations that hinder analysis, avoiding black-box ``over-restoration.'' These results establish PRISM as a generalizable and controllable framework for high-fidelity restoration in domains where scientific utility is a priority.

---
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://prismrestore.github.io/)
[![Demo](https://img.shields.io/badge/Data-Download-green)](https://drive.google.com/drive/folders/19VNlF2O3F5axlRoRSlIh-rFi5jmHmk0N?usp=sharing)
[![Weights](https://img.shields.io/badge/Model-Weights-orange)](https://drive.google.com/drive/folders/124vCNRlQuOCnO6SkwZySfMJf-m261-zR?usp=sharing)

---

**PRISM** is a novel compositional approach to image restoration that handles multiple degradations simultaneously through contrastive disentanglement and compound-aware supervision. Unlike traditional methods that train on single distortions, PRISM learns from full combinatorial mixture sets, enabling superior performance on compound degradations commonly found in real-world scenarios.

## Installation

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

3. **Install additional packages for evaluation**:
```bash
pip install lpips  # For perceptual metrics
pip install scikit-image opencv-python matplotlib
pip install pandas seaborn  # For analysis and visualization
```

## Data and Weights

### Pre-trained Weights
Download the pre-trained PRISM model weights from [here](https://drive.google.com/drive/folders/124vCNRlQuOCnO6SkwZySfMJf-m261-zR?usp=sharing) and save them to the `pre-trained` folder.

### Training and Evaluation Data
Download the training and evaluation datasets from [here](https://drive.google.com/drive/folders/19VNlF2O3F5axlRoRSlIh-rFi5jmHmk0N?usp=sharing) and save them to the `data` folder.

Download from here: .

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
To run inference on a single image, run
```bash
# Run inference on a single image
python infer.py --input path/to/degraded/image.jpg --output path/to/restored/image.jpg --model weights/prism_model.pth

# Batch inference
python infer.py --input_dir path/to/degraded/images/ --output_dir path/to/restored/images/ --model weights/prism_model.pth
```
You can also use the bash script.
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
```

## Evaluation

### Single Image Inference

Run inference on a single image with a natural language prompt:

```bash
python infer.py         
    --prism_checkpoint_path /path/to/model/checkpoint         
    --distortion_type task        
    --img_path /path/to/test/image         
    --save_root /path/to/output         
    --num_inference_steps 20         
    --seed 42
```

For more examples, see `infer.sh` which provides a convenient wrapper script for common inference tasks.

### Batch Evaluation with Metrics

The `run_eval.py` script runs inference over entire test datasets and computes quantitative metrics (PSNR, SSIM, LPIPS, FID):

```bash
python run_eval.py \
    --input_dir /path/to/test/images \
    --results_dir /path/to/model/checkpoint \
    --output_dir /path/to/output \
    --num_inference_steps 20 \
    --seed 42
```

This will:
1. Process all images in the input directory
2. Save restored outputs to the output directory
3. Compute and save metrics (PSNR, SSIM, LPIPS, FID) to `all_metrics.json`
4. Print a summary table with results for each degradation type

### Downstream Task Evaluation

We provide Jupyter notebooks for evaluating PRISM on domain-specific downstream tasks in the `downstream_evaluation/` directory:

- **`downstream_eval_microscopy.ipynb`** - Cell segmentation and counting tasks on microscopy images
- **`downstream_eval_satellite.ipynb`** - Land cover classification and change detection on satellite imagery
- **`downstream_eval_species.ipynb`** - Species classification on wildlife monitoring data
- **`downstream_eval_urban.ipynb`** - Object detection and scene understanding on urban imagery

Each notebook demonstrates how restoration quality impacts task performance compared to using degraded images directly.

<!-- ## Interactive Demos

The `demo.ipynb` notebook provides a comprehensive, step-by-step exploration of PRISM.

### Run this Notebook
```bash
# Start Jupyter notebook
jupyter notebook demo.ipynb
``` -->

## Test it out yourself!

We also provide a Gradio demo for running a few examples on your own! Simply run 
```bash
python app.py
```
and then open http://localhost:7860 to test the model. We provide some example inputs in `data/demo.` We also provide more examples from our test dataset in **COME BACK TO THIS**.

## Baselines

We compare PRISM against state-of-the-art methods across three categories:

- **Encoder-Decoder Backbones**: AirNet, Restormer, NAFNet
- **Multi-Degradation Methods**: OneRestore, PromptIR  
- **Modular/Token-Based Methods**: DiffPlugin, MPerceiver, AutoDIR

See [baselines.md](baselines.md) for detailed information on downloading, installing, and retraining baseline methods.

<!-- ## Citation

If you find PRISM useful in your research, please cite our paper!

```bibtex
@inproceedings{prism2024,
  title={PRISM: A Compositional Approach to Image Restoration},
  author={[Author Names]},
  booktitle={[Conference/Journal Name]},
  year={2024}
}
``` -->