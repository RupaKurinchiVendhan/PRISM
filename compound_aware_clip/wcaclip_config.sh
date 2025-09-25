# WCACLIP Training Configuration

# Data Configuration
DATA_ROOT="data/train"
CLEAN_DIR_NAME="clear"
OUTPUT_DIR="./wcaclip_results"

# Model Configuration
CLIP_MODEL="openai/clip-vit-large-patch14"
FREEZE_TEXT_ENCODER=false

# Training Configuration
BATCH_SIZE=16
LEARNING_RATE=1e-5
NUM_EPOCHS=10
WARMUP_STEPS=500
TEMPERATURE=0.07
QUALITY_LOSS_WEIGHT=1.0

# System Configuration
MIXED_PRECISION="fp16"
NUM_WORKERS=4
SEED=42

# Logging Configuration
LOGGING_STEPS=50
SAVE_STEPS=1000
EVAL_STEPS=500