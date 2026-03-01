#!/bin/bash

#SBATCH --job-name=eval    # Updated name for partial restoration training
#SBATCH -o /data/vision/beery/scratch/rupa/image-restoration/Diff-Plugin2/eval_all_%j.log
#SBATCH --qos=vision-beery-main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512GB
#SBATCH --time=7-00:00:00
#SBATCH --account=vision-beery
#SBATCH --partition=vision-beery


source /data/vision/beery/scratch/rupa/anaconda3/etc/profile.d/conda.sh 
conda activate prism

# Batch eval script for all degradation types
# Usage: bash run_eval.sh

INPUT_DIR="/data/vision/beery/scratch/rupa/image-restoration/OneRestore/mdb_benchmark"
RESULTS_DIR="/data/vision/beery/scratch/rupa/image-restoration/Diff-Plugin2/results_all"
OUTPUT_DIR="/data/vision/beery/scratch/rupa/image-restoration/Diff-Plugin2/eval_outputs"

# Install required packages if needed
echo "Checking for required packages..."
pip install -q torchvision torchmetrics pillow tqdm transformers diffusers
pip install scikit-image opencv-python

# Run the batch eval script
python3 run_eval.py \
    --input_dir=$INPUT_DIR \
    --results_dir=$RESULTS_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=512 \
    --num_eval_steps=20 \
    --seed=42

# python3 run_eval.py --skip_eval

echo ""
echo "Evaluation complete! Results saved to: $OUTPUT_DIR"
echo "Metrics saved to: $OUTPUT_DIR/all_metrics.json"
