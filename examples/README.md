# AttnDiff Examples

## CLI Usage

### Compute Fingerprint

```bash
# Using pre-extracted attention files
uv run attndiff-compute \
  --original output/attention/model_att_origin.json \
  --corrupted output/attention/model_att_perturb.json \
  --out output/comput_W/fingerprint_model.json

# Extract attention and compute fingerprint
uv run attndiff-compute \
  --model_name meta-llama/Llama-2-7b-hf \
  --attn_device cuda:0 \
  --dataset dataset/dataset.json \
  --out output/comput_W/fingerprint_llama2.json
```

### Compare Fingerprints

```bash
# Compare all fingerprints in a directory
uv run attndiff-compare \
  --base output/comput_W/fingerprint_llama2.json \
  --dir output/comput_W \
  --cka linear

# Compare specific layer
uv run attndiff-compare \
  --base output/comput_W/fingerprint_llama2.json \
  --dir output/comput_W \
  --cka linear \
  --layer 1
```

## Dataset Format

Create a dataset file `dataset/dataset.json`:

```json
[
  {
    "id": 1,
    "topic": "Mathematics",
    "original": "What is 2 + 2?",
    "corrupted": "Waht is 2 + 2?"
  },
  {
    "id": 2,
    "topic": "Science",
    "original": "Explain photosynthesis.",
    "corrupted": "Explan photosynthesis."
  }
]
```
