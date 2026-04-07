#!/bin/bash
set -euo pipefail

# Batch compute fingerprints for multiple models
# Usage: bash scripts/batch_compute.sh

# Configuration
ATTN_DEVICE="cuda:6"
MODE="diff"
DATASET="dataset/dataset.json"

# Model paths mapping
declare -A MODEL_PATHS
MODEL_PATHS["Llama-2-7B"]="path/to/llama-2-7b"
MODEL_PATHS["Qwen2.5-7B"]="/data/models/Qwen2-7B-Instruct"
MODEL_PATHS["WizardMath-7B"]="path/to/wizardmath-7b"
# Add more models here

# Models to process
MODELS=(
    #"Llama-2-7B"
    "Qwen2.5-7B"
    # "WizardMath-7B"
)

echo "Starting batch fingerprint computation..."
echo "Device: $ATTN_DEVICE"
echo "Mode: $MODE"
echo ""

for model_name in "${MODELS[@]}"; do
    if [[ -v MODEL_PATHS["$model_name"] ]]; then
        model_path="${MODEL_PATHS[$model_name]}"
        output_file="output/comput_W/fingerprint_${model_name}.json"
        
        echo "========================================="
        echo "Processing: $model_name"
        echo "Path: $model_path"
        echo "Output: $output_file"
        echo "========================================="
        
        uv run attndiff-compute \
            --model_name "$model_path" \
            --attn_device "$ATTN_DEVICE" \
            --mode "$MODE" \
            --out "$output_file"
        
        echo "✓ Completed: $model_name"
        echo ""
    else
        echo "⚠ Warning: Model '$model_name' not found in MODEL_PATHS"
    fi
done

echo "========================================="
echo "Batch computation completed!"
echo "========================================="
