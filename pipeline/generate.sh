set -euo pipefail

# Mapping of model names to their paths
declare -A MODEL_PATHS

BASE_PATH="your_path_to_modelmap"

# example:
MODEL_PATHS["chinese-llama-2-7b"]="$BASE_PATH/hfl/chinese-llama-2-7b"
MODEL_PATHS["llemma_7b"]="$BASE_PATH/EleutherAI/llemma_7b"

MODELS=(
    #"gemma-2-2B-it-4Bit-GPTQ"
    #"gemma-2-2b"
    #"Mistral-7B-Instruct-v0.3-GPTQ-4bit"
)

ATTN_DEVICE="cuda:6"
for model_name in "${MODELS[@]}"; do
    if [[ -v MODEL_PATHS["$model_name"] ]]; then
        echo "Processing $model_name..."
        
        CURRENT_DEVICE="$ATTN_DEVICE"

        python compute_W.py \
          --model_name "${MODEL_PATHS[$model_name]}" \
          --original "output/attention/${model_name}_att_origin.json" \
          --corrupted "output/attention/${model_name}_att_perturb.json" \
          --mode diff \
          --attn_device "$CURRENT_DEVICE" \
          --out output/comput_W/fingerprint_${model_name}.json

        echo "Finished $model_name"
    else
        echo "Error: Model '$model_name' not found in mapping table."
    fi
done