# Model Pruning Tools

Tools for structured and unstructured model pruning.

## Setup

### 1. Clone LLM-Pruner Framework

```bash
cd tools/model-pruning
git clone https://github.com/horseee/LLM-Pruner.git
```

### 2. Install Dependencies

```bash
# Create conda environment (recommended)
conda env create -f model-pruning.yaml
conda activate model-pruning

# Or install with pip
cd LLM-Pruner
pip install -r requirements.txt
```

## Pruning Strategies

We support four different pruning strategies:
- **Taylor**: Taylor expansion-based importance
- **Magnitude**: Weight magnitude-based pruning
- **Random**: Random pruning (baseline)
- **Wanda**: Structured pruning method

## Usage

Place your pruning scripts (e.g., `get_pruned_models.sh`) in this directory, then:

```bash
cd tools/model-pruning
bash get_pruned_models.sh
```

The script will generate pruned models in the target model directory:
```
your_model/
└── prune/
    ├── taylor-0.10/
    ├── magnitude-0.10/
    ├── random-0.10/
    └── wanda-0.10/
```

## Directory Structure

```
tools/model-pruning/
├── LLM-Pruner/              # LLM-Pruner framework (clone from GitHub)
├── get_pruned_models.sh     # Your pruning script
├── model-pruning.yaml       # Conda environment config
└── README.md                # This file
```

## References

- [LLM-Pruner GitHub](https://github.com/horseee/LLM-Pruner)
- [Structured Pruning Paper](https://arxiv.org/abs/1608.08710)
- [Attention Head Pruning](https://arxiv.org/abs/1905.10650)
