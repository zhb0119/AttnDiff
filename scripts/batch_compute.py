#!/usr/bin/env python
"""
Batch compute fingerprints for multiple models.
Usage: uv run python scripts/batch_compute.py
"""

import subprocess

# Configuration
ATTN_DEVICE = "cuda:0"
MODE = "diff"
DATASET = "dataset/dataset.json"

# Model paths mapping
MODEL_PATHS = {
    "Llama-2-7B": "path/to/llama-2-7b",
    "Qwen2.5-7B": "path/to/qwen2.5-7b",
    "WizardMath-7B": "path/to/wizardmath-7b",
    # Add more models here
}

# Models to process
MODELS = [
    "Llama-2-7B",
    "Qwen2.5-7B",
    # "WizardMath-7B",
]


def main():
    print("Starting batch fingerprint computation...")
    print(f"Device: {ATTN_DEVICE}")
    print(f"Mode: {MODE}")
    print()

    for model_name in MODELS:
        if model_name not in MODEL_PATHS:
            print(f"⚠ Warning: Model '{model_name}' not found in MODEL_PATHS")
            continue

        model_path = MODEL_PATHS[model_name]
        output_file = f"output/comput_W/fingerprint_{model_name}.json"

        print("=" * 60)
        print(f"Processing: {model_name}")
        print(f"Path: {model_path}")
        print(f"Output: {output_file}")
        print("=" * 60)

        try:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "attndiff-compute",
                    "--model_name",
                    model_path,
                    "--attn_device",
                    ATTN_DEVICE,
                    "--dataset",
                    DATASET,
                    "--mode",
                    MODE,
                    "--out",
                    output_file,
                ],
                check=True,
            )
            print(f"✓ Completed: {model_name}")
            print()
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {model_name}")
            print(f"Error: {e}")
            print()

    print("=" * 60)
    print("Batch computation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
