# Dataset Directory

Place your dataset files here.

## Format

Create a `dataset.json` file with the following structure:

```json
[
  {
    "id": 1,
    "topic": "Category",
    "original": "Original prompt text",
    "corrupted": "Corrupted version of the prompt"
  }
]
```

Each entry should contain:
- `id`: Unique identifier
- `topic`: Category or topic (optional)
- `original`: The original prompt
- `corrupted`: A perturbed version of the original prompt

## Purpose

The dataset is used to extract attention patterns from models under paired conditions (original vs. corrupted prompts), which forms the basis for fingerprint computation.
