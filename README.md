<div align="center">

# AttnDiff: Attention-based Differential Fingerprinting for Large Language Models

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource&logoColor=white)](LICENSE)

</div>

## Introduction

AttnDiff is a lightweight model fingerprinting method for **model similarity estimation**. Instead of comparing hidden states, AttnDiff builds a fingerprint from **head-level attention differences** under paired prompts (e.g., `original` vs. `corrupted`).

### Pipeline Overview

<div align="center">
<img src="figure/pipeline.png" width="900" alt="AttnDiff pipeline" />
<img src="figure/pool.png" width="900" alt="AttnDiff pool" />
</div>

## Quick Start

### Installation

**Using uv (Recommended)**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/zhb0119/AttnDiff.git
cd AttnDiff
uv sync
```

### Basic Usage

**Compute fingerprints:**

```bash
# Edit scripts/batch_compute.sh to configure models and device
bash scripts/batch_compute.sh
```

**Compare fingerprints:**

You can use pre-computed fingerprints provided in the repository:

```bash
uv run attndiff-compare \
  --base output/comput_W/fingerprint_Llama-2-7B.json \
  --dir output/comput_W \
  --cka linear
```

> **Note**: The repository includes pre-computed fingerprints for several open-source models in `output/comput_W/`. You can use these to quickly test the comparison functionality without computing fingerprints yourself.

## Table of Contents

- [AttnDiff: Attention-based Differential Fingerprinting for Large Language Models](#attndiff-attention-based-differential-fingerprinting-for-large-language-models)
  - [Introduction](#introduction)
    - [Pipeline Overview](#pipeline-overview)
  - [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Basic Usage](#basic-usage)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
    - [Compute Fingerprints](#compute-fingerprints)
    - [Compare Fingerprints](#compare-fingerprints)
  - [Dataset Format](#dataset-format)
  - [Repository Structure](#repository-structure)
  - [Partial Model List](#partial-model-list)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Usage

### Compute Fingerprints

**Recommended: Use batch script**

```bash
# Edit scripts/batch_compute.sh to configure model paths and device
bash scripts/batch_compute.sh
```

**Advanced: Manual computation**

```bash
# From pre-extracted attention files
uv run attndiff-compute \
  --original output/attention/model_att_origin.json \
  --corrupted output/attention/model_att_perturb.json \
  --mode diff \
  --out output/comput_W/fingerprint_model.json

# Or let the tool auto-extract attentions from model
uv run attndiff-compute \
  --model_name /path/to/your/model \
  --attn_device cuda:0 \
  --mode diff \
  --out output/comput_W/fingerprint_your_model.json
```

**Arguments:**
- `--model_name`: Model name or local path
- `--original`: Path to original attention JSON
- `--corrupted`: Path to corrupted attention JSON
- `--mode`: `diff` (default), `orig`, or `base`
- `--attn_device`: Device for attention extraction (e.g., `cuda:0`)
- `--out`: Output fingerprint path

### Compare Fingerprints

```bash
# Compare all fingerprints in directory
uv run attndiff-compare \
  --base output/comput_W/fingerprint_base.json \
  --dir output/comput_W \
  --cka linear

# Compare specific layer
uv run attndiff-compare \
  --base output/comput_W/fingerprint_Llama-2-7B.json \
  --dir output/comput_W \
  --cka linear \
  --layer 1
```

**Arguments:**
- `--base`: Base fingerprint JSON (required)
- `--dir`: Directory containing fingerprints
- `--cka`: CKA type (`linear`)
- `--layer`: Compare specific layer (1-based, optional)

## Dataset Format

Create `dataset/dataset.json`:

```json
[
  {
    "id": 1,
    "topic": "Mathematics",
    "original": "...",
    "corrupted": "..."
  },
  {
    "id": 2,
    "topic": "Science",
    "original": "...",
    "corrupted": "..."
  }
]
```

## Repository Structure

```
AttnDiff/
├── src/attndiff/          # Package source code
│   ├── core/              # Core algorithms
│   ├── cli/               # CLI tools
│   └── utils/             # Utilities
├── tools/                 # Model manipulation tools
│   ├── model-merging/     # Model merging tools
│   └── model-pruning/     # Model pruning tools
├── scripts/               # Batch processing scripts
├── tests/                 # Unit tests
├── examples/              # Usage examples
├── dataset/               # Dataset directory
├── output/                # Output directory
│   ├── attention/         # Attention files
│   └── comput_W/          # Fingerprints
├── pyproject.toml         # UV/pip configuration
└── README.md
```

## Partial Model List

| ID | Model Name | Repository URL |
|----|------------|----------------|
| 1 | Llama-2-7b-ppo-lora | [huggingface.co/renyiyu/llama-2-7b-ppo-lora-v0.1](https://huggingface.co/renyiyu/llama-2-7b-ppo-lora-v0.1) |
| 2 | Tulu-2-dpo-7b | [huggingface.co/allenai/tulu-2-dpo-7b](https://huggingface.co/allenai/tulu-2-dpo-7b) |
| 3 | Llama2-7b-dpo | [huggingface.co/mncai/llama2-7b-dpo-v1](https://huggingface.co/mncai/llama2-7b-dpo-v1) |
| 4 | Qwen2.5-Coder-1.5B | [huggingface.co/Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) |
| 5 | Qwen2.5-Math-1.5B | [huggingface.co/Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) |
| 6 | Qwen2.5-1.5B-Instruct | [huggingface.co/Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| 7 | Qwen2.5-14B-Instruct | [huggingface.co/Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| 8 | Oxy-1-small | [huggingface.co/oxyapi/oxy-1-small](https://huggingface.co/oxyapi/oxy-1-small) |
| 9 | Qwen2.5-14B-Gutenberg-Instruct-Slerpeno | [huggingface.co/v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno](https://huggingface.co/v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno) |
| 10 | Gemma-2-2b-neogenesis-ita | [huggingface.co/anakin87/gemma-2-2b-neogenesis-ita](https://huggingface.co/anakin87/gemma-2-2b-neogenesis-ita) |
| 11 | Gemma-2-baku-2b | [huggingface.co/rinna/gemma-2-baku-2b](https://huggingface.co/rinna/gemma-2-baku-2b) |
| 12 | Gemma2-2b-merged | [huggingface.co/vonjack/gemma2-2b-merged](https://huggingface.co/vonjack/gemma2-2b-merged) |
| 13 | AQUA-7B | [huggingface.co/KurmaAI/AQUA-7B](https://huggingface.co/KurmaAI/AQUA-7B) |
| 14 | Spellcheck-mistral-7b | [huggingface.co/openfoodfacts/spellcheck-mistral-7b](https://huggingface.co/openfoodfacts/spellcheck-mistral-7b) |
| 15 | Mistral-7B-Instruct-demi-merge | [huggingface.co/grimjim/Mistral-7B-Instruct-demi-merge-v0.3-7B](https://huggingface.co/grimjim/Mistral-7B-Instruct-demi-merge-v0.3-7B) |
| 16 | Llama-3.1-8B-Instruct-Open-R1-Distill | [huggingface.co/asas-ai/Llama-3.1-8B-Instruct-Open-R1-Distill](https://huggingface.co/asas-ai/Llama-3.1-8B-Instruct-Open-R1-Distill) |
| 17 | Qwen2.5-7B-Open-R1-Distill | [huggingface.co/erickrus/Qwen2.5-7B-Open-R1-Distill](https://huggingface.co/erickrus/Qwen2.5-7B-Open-R1-Distill) |
| 18 | DeepSeek-R1-Distill-Qwen-14B | [huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| 19 | Llama-2-7b-logit-watermark-distill | [huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2) |
| 20 | Instruct_Mixtral-8x7B-Dolly15K | [huggingface.co/Brillibits/Instruct_Mixtral-8x7B-v0.1_Dolly15K](https://huggingface.co/Brillibits/Instruct_Mixtral-8x7B-v0.1_Dolly15K) |
| 21 | Nous-Hermes-2-Mixtral-8x7B-DPO | [huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) |
| 22 | Openbuddy-mixtral-8x7b-v15.4 | [huggingface.co/openbuddy/openbuddy-mixtral-8x7b-v15.4](https://huggingface.co/openbuddy/openbuddy-mixtral-8x7b-v15.4) |
| 23 | Qwen2.5-7B-Instruct-GPTQ-Int8 | [huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8) |
| 24 | Qwen2.5-7B-Instruct-GPTQ-Int4 | [huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4) |
| 25 | Llama-2-7B-Chat-GPTQ | [huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) |
| 26 | Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 | [huggingface.co/iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8](https://huggingface.co/iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8) |
| 27 | LLaMA-3.1-8B-Instruct-INT4-GPTQ | [huggingface.co/DaraV/LLaMA-3.1-8B-Instruct-INT4-GPTQ](https://huggingface.co/DaraV/LLaMA-3.1-8B-Instruct-INT4-GPTQ) |
| 28 | Mistral-7B-Instruct-v0.3-GPTQ-4bit | [huggingface.co/RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit](https://huggingface.co/RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit) |

## Citation

If you use AttnDiff in your research, please cite:

```bibtex
coming soon
```

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Uses [Hugging Face Transformers](https://huggingface.co/transformers)
- Package management with [UV](https://github.com/astral-sh/uv)
