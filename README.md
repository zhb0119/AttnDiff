<div align="center">

# AttnDiff: Attention-based Differential Fingerprinting for Large Language Models

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource&logoColor=white)](LICENSE)
[![CI](https://github.com/yourusername/AttnDiff/workflows/CI/badge.svg)](https://github.com/yourusername/AttnDiff/actions)

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

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Repository Structure](#repository-structure)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

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
