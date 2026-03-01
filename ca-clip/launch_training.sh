#!/bin/bash
# Launch script for CA-CLIP training

set -e  # Exit on error

echo "=========================================="
echo "CA-CLIP Training Launcher"
echo "=========================================="
echo ""

# Parse arguments
GPU_IDS="0"
CONFIG="configs/train.yml"
NUM_GPUS=1
DISTRIBUTED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPU_IDS="$2"
            NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus GPU_IDS       GPU IDs to use (e.g., '0' or '0,1,2,3')"
            echo "  --config CONFIG      Path to config file (default: configs/train.yml)"
            echo "  --distributed        Enable distributed training (auto-enabled for multi-GPU)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  Single GPU:    $0 --gpus 0"
            echo "  Multi GPU:     $0 --gpus 0,1,2,3"
            echo "  Custom config: $0 --gpus 0 --config configs/my_train.yml"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-enable distributed for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    DISTRIBUTED=true
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Check if training script exists
if [ ! -f "ca_clip/train.py" ]; then
    echo "ERROR: Training script not found: ca_clip/train.py"
    echo "Make sure you're in the ca-clip-package directory"
    exit 1
fi

# Display configuration
echo "Configuration:"
echo "  GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo "  Config: $CONFIG"
echo "  Distributed: $DISTRIBUTED"
echo ""

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Launch training
if [ "$DISTRIBUTED" = true ]; then
    echo "Launching distributed training on $NUM_GPUS GPUs..."
    echo ""
    
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=4321 \
        -m ca_clip.train \
        -opt=$CONFIG \
        --launcher pytorch
else
    echo "Launching single GPU training..."
    echo ""
    
    python -m ca_clip.train \
        -opt=$CONFIG \
        --launcher none
fi

echo ""
echo "=========================================="
echo "Training finished!"
echo "=========================================="
