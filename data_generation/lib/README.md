# Weather and Cloud Distortion Augmentation Tools

This guide documents how to use third-party open-source scripts to apply synthetic **rain**, **snow**, and **cloud cover** to images, primarily for training and evaluating models on degraded input scenarios.

---

## External Repositories Used

### 1. **Weather Effect Generator (Rain & Snow)**
- **Source**: https://github.com/hgupta01/Weather_Effect_Generator
- **Description**: This repository provides Python scripts to apply realistic **rain** and **snow** effects to images.
- **Citation**: If used in academic work, please cite the original authors as linked in the repository.

### 2. **Satellite Cloud Generator**
- **Source**: https://github.com/strath-ai/SatelliteCloudGenerator
- **Description**: This repository contains a PyTorch-based tool for overlaying synthetic **cloud cover** on satellite or aerial images. Useful for remote sensing and cloud occlusion benchmarks.
- **Citation**: See `LICENSE` and publication information in the repository for proper attribution.

---

## Setup

### Clone the Repositories

```bash
git clone https://github.com/hgupta01/Weather_Effect_Generator.git
git clone https://github.com/strath-ai/SatelliteCloudGenerator.git
````

### Install Dependencies

Each repository contains a `requirements.txt` file or environment setup instructions. Install dependencies (ideally in a virtual environment) using:

```bash
cd Weather_Effect_Generator
pip install -r requirements.txt

cd ../SatelliteCloudGenerator
pip install -r requirements.txt
```

---

## 💧 Applying Rain and Snow

The `Weather_Effect_Generator` provides standalone scripts for generating rain and snow overlays for images. You can batch-process an entire folder by adapting these scripts. Control parameters such as intensity and particle size by editing script arguments or config files.

---

## Adding Synthetic Cloud Cover

The `SatelliteCloudGenerator` generates cloud masks and image overlays. Masks are sampled and warped before being overlaid. Parameters such as cloud coverage, brightness, and randomness can be tuned via command-line flags.

---

## Output Format

Both scripts overwrite or save new images in your specified output directory. Ensure:

* You preserve original image structure or filenames for consistency.
* You optionally store metadata about the applied transformation for later use.

---

## Integration Notes

* These augmentations are useful for generating training or evaluation data for models like PRISM.
* For PRISM, append a label or prompt (e.g., `"remove clouds"`, `"remove snow"`) describing the transformation applied to the image.
* These augmentations can be integrated into a larger pipeline for **composite degradation** by sequentially applying multiple distortions.