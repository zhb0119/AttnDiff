# Model Merging Tools

Tools for merging multiple models or model weights.

## Setup

### 1. Clone Mergekit Framework

```bash
cd tools/model-merging
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
```

### 2. Install Dependencies

```bash
# Create conda environment (recommended)
conda env create -f model-merging.yaml
conda activate model-merging
```

## Merging Strategies

We support four different merging strategies:

1. **Task-based Merging**: Weighted average based on task performance
2. **Task-based with DARE**: Task merging with drop and rescale
3. **TIES Merging**: Trim, elect, and merge based on parameter importance
4. **TIES with DARE**: TIES combined with DARE for better stability

## Usage

Place your merging scripts (e.g., `batch-mergekit.py`) in this directory, then:

```bash
cd tools/model-merging
python batch-mergekit.py
```

The script will generate merged models in subdirectories:
```
your_model/
├── task/              # Task-based merging results
├── dare_task/         # Task + DARE results
├── ties/              # TIES merging results
└── dare_ties/         # TIES + DARE results
```

## Directory Structure

```
tools/model-merging/
├── mergekit/                # Mergekit framework (clone from GitHub)
├── batch-mergekit.py        # Your merging script
├── model-merging.yaml       # Conda environment config
└── README.md                # This file
```

## Merging Configuration Example

```yaml
# Example mergekit config
models:
  - model: model_a
    parameters:
      weight: 0.5
  - model: model_b
    parameters:
      weight: 0.5
merge_method: linear
dtype: float16
```

## References

- [Mergekit GitHub](https://github.com/arcee-ai/mergekit)
- [Model Merging Survey](https://arxiv.org/abs/2403.01187)
- [TIES-Merging](https://arxiv.org/abs/2306.01708)
- [DARE](https://arxiv.org/abs/2311.03099)
