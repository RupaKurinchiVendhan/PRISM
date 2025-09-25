#!/bin/bash

# WCACLIP Training Launch Script
# This script sets up the environment and launches WCACLIP training

set -e  # Exit on any error

echo "🚀 Starting WCACLIP Training Pipeline"
echo "======================================"

# Source configuration
if [ -f "./wcaclip_config.sh" ]; then
    source ./wcaclip_config.sh
    echo "Configuration loaded from wcaclip_config.sh"
else
    echo "⚠️  Warning: wcaclip_config.sh not found, using default values"
    # Default configuration
    DATA_ROOT="data/train"
    CLEAN_DIR_NAME="clear"
    OUTPUT_DIR="./wcaclip_results"
    CLIP_MODEL="openai/clip-vit-large-patch14"
    FREEZE_TEXT_ENCODER=false
    BATCH_SIZE=1
    LEARNING_RATE=1e-5
    NUM_EPOCHS=10
    WARMUP_STEPS=500
    TEMPERATURE=0.07
    QUALITY_LOSS_WEIGHT=1.0
    MIXED_PRECISION="fp16"
    NUM_WORKERS=4
    SEED=42
    LOGGING_STEPS=50
    SAVE_STEPS=1000
    EVAL_STEPS=500
fi

# Check Python environment
echo "🔍 Checking Python environment..."
python --version

# Check if required packages are installed
echo "📦 Checking required packages..."
REQUIRED_PACKAGES=("torch" "transformers" "accelerate" "pandas" "numpy" "pillow")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "❌ Missing required packages: ${MISSING_PACKAGES[*]}"
    echo "Installing missing packages..."
    pip install "${MISSING_PACKAGES[@]}"
fi

# Check GPU availability
echo "🖥️  Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Create output directory
echo "📁 Creating output directory..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

# Check if training data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Error: Training data directory not found: $DATA_ROOT"
    echo "Please prepare training data first using prepare_wcaclip_data.py"
    exit 1
fi

# Prepare training data if needed
CONTRASTIVE_DATA_PATH="./data/contrastive/train_contrastive"
if [ ! -f "$CONTRASTIVE_DATA_PATH" ]; then
    echo "📊 Preparing WCACLIP training data..."
    python prepare_wcaclip_data.py \
        --data_root "$DATA_ROOT" \
        --clean_dir_name "$CLEAN_DIR_NAME" \
        --output_dir "$CONTRASTIVE_DATA_PATH" \
        --min_images_per_degradation 10
else
    echo "Training data already prepared: $CONTRASTIVE_DATA_PATH"
fi

# Launch training
echo "🎯 Launching WCACLIP training..."
echo "Configuration:"
echo "  - Data root: $DATA_ROOT"
echo "  - Output dir: $OUTPUT_DIR"
echo "  - CLIP model: $CLIP_MODEL"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Mixed precision: $MIXED_PRECISION"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/logs/training_$TIMESTAMP.log"

# Launch training with all parameters
# Build command with conditional freeze_text_encoder flag

CONTRASTIVE_DATA_PATH="./data/contrastive/train_contrastive/wcaclip_train.csv"
TRAIN_CMD="python train_wcaclip.py \
    --data_csv \"$CONTRASTIVE_DATA_PATH\" \
    --distortion_taxonomy \"./data/contrastive/train_contrastive/distortion_taxonomy.json\" \
    --data_root \"$DATA_ROOT\" \
    --output_dir \"$OUTPUT_DIR\" \
    --clip_model_name \"$CLIP_MODEL\""

# Add freeze_text_encoder flag only if set to true
if [ "$FREEZE_TEXT_ENCODER" = "true" ] || [ "$FREEZE_TEXT_ENCODER" = "True" ] || [ "$FREEZE_TEXT_ENCODER" = "TRUE" ]; then
    TRAIN_CMD="$TRAIN_CMD --freeze_text_encoder"
fi

TRAIN_CMD="$TRAIN_CMD \
    --train_batch_size \"$BATCH_SIZE\" \
    --learning_rate \"$LEARNING_RATE\" \
    --num_train_epochs \"$NUM_EPOCHS\" \
    --warmup_steps \"$WARMUP_STEPS\" \
    --temperature \"$TEMPERATURE\" \
    --quality_loss_weight \"$QUALITY_LOSS_WEIGHT\" \
    --mixed_precision \"$MIXED_PRECISION\" \
    --dataloader_num_workers \"$NUM_WORKERS\" \
    --seed \"$SEED\" \
    --logging_steps \"$LOGGING_STEPS\" \
    --save_steps \"$SAVE_STEPS\" \
    --eval_steps \"$EVAL_STEPS\""

# Execute the command
eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 WCACLIP training completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "📝 Training log: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate the model: python evaluate_wcaclip.py --model_path $OUTPUT_DIR/final_model"
    echo "  2. Check training logs: cat $LOG_FILE"
    echo "  3. Visualize results in $OUTPUT_DIR"
else
    echo ""
    echo "❌ WCACLIP training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📝 Check the log file for details: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi