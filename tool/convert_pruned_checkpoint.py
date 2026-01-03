#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert an LLM-Pruner style checkpoint (torch.save({'model': model, 'tokenizer': tokenizer}, ...))
into a standard HuggingFace directory layout that can be used with
`from_pretrained` and your existing fingerprint pipeline.

Usage:
  python convert_pruned_checkpoint.py \
    --ckpt /path/to/random-0.10/2025-12-07-16-56-12/pytorch_model.bin \
    --out_dir /home/kdz/data/OpenSourceModels/meta-llama/Llama-2-finance-7B-random-0.10

After conversion, set in generate.sh:
  MODEL_PATHS["Llama-2-finance-7B-random-0.10"]="/home/kdz/data/OpenSourceModels/meta-llama/Llama-2-finance-7B-random-0.10"
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LLM-Pruner checkpoint (pytorch_model.bin with model+tokenizer) "
        "to a HuggingFace directory layout."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to LLM-Pruner checkpoint file (pytorch_model.bin).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Target directory to save HF-style model and tokenizer.",
    )
    parser.add_argument(
        "--llmpruner_root",
        type=str,
        default="/home/kdz/data/xzh/zhb/EverTracer-main/EverTracer-main/Experiments/model-pruning/LLM-Pruner-main",
        help=(
            "Path to local LLM-Pruner-main repo (containing the LLMPruner package). "
            "This is needed so torch.load can import LLMPruner.* classes when "
            "unpickling the checkpoint. If you moved the repo, override this path."
        ),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="If set, cast model to half precision before saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the LLMPruner package used when saving the checkpoint is importable
    llmpruner_root = Path(args.llmpruner_root)
    if llmpruner_root.exists():
        llmpruner_root_str = str(llmpruner_root.resolve())
        if llmpruner_root_str not in sys.path:
            sys.path.insert(0, llmpruner_root_str)
        try:
            __import__("LLMPruner")  # noqa: F401 - ensure module is importable
        except ImportError:
            print(
                f"[convert] Warning: LLMPruner package not importable from {llmpruner_root_str}. "
                "If you see ModuleNotFoundError: 'LLMPruner', please check --llmpruner_root."
            )
    else:
        print(
            f"[convert] Warning: --llmpruner_root path does not exist: {llmpruner_root}. "
            "If torch.load fails with ModuleNotFoundError('LLMPruner'), please set the correct path."
        )

    print(f"[convert] Loading checkpoint from: {ckpt_path}")
    # PyTorch 2.6+ defaults to weights_only=True, which prevents loading
    # arbitrary Python objects stored by torch.save. Here the checkpoint is
    # created locally by LLM-Pruner and is trusted, so we explicitly set
    # weights_only=False to allow loading the full dict with model+tokenizer.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict with keys 'model' and 'tokenizer'.")

    model = ckpt.get("model")
    tokenizer = ckpt.get("tokenizer")

    if model is None or tokenizer is None:
        raise ValueError(
            "Checkpoint dict must contain 'model' and 'tokenizer' keys. "
            f"Found keys: {list(ckpt.keys())}"
        )

    # Try to align config.intermediate_size with the pruned MLP width so that
    # HuggingFace's from_pretrained does not see Linear weight size mismatches
    # such as [4096, 9907] vs [4096, 11008]. We inspect the first layer's
    # MLP gate_proj to infer the pruned intermediate size.
    try:
        core_model = getattr(model, "model", None)
        layers = getattr(core_model, "layers", None)
        if layers and len(layers) > 0:
            layer0 = layers[0]
            mlp = getattr(layer0, "mlp", None)
            gate_proj = getattr(mlp, "gate_proj", None)
            if gate_proj is not None and hasattr(gate_proj, "weight"):
                pruned_intermediate = gate_proj.weight.shape[0]
                cfg = getattr(model, "config", None)
                if cfg is not None:
                    current_intermediate = getattr(cfg, "intermediate_size", None)
                    if current_intermediate != pruned_intermediate:
                        print(
                            f"[convert] Adjusting config.intermediate_size "
                            f"from {current_intermediate} to {pruned_intermediate} "
                            "to match pruned MLP width."
                        )
                        cfg.intermediate_size = pruned_intermediate
    except Exception as e:
        print(f"[convert] Warning: failed to adjust config.intermediate_size automatically: {e}")

    if args.half and hasattr(model, "half"):
        print("[convert] Casting model to half precision (fp16)...")
        model = model.half()

    # Move to CPU before saving to avoid device-specific issues
    if hasattr(model, "to"):
        model = model.to("cpu")

    print(f"[convert] Saving tokenizer to: {out_dir}")
    tokenizer.save_pretrained(str(out_dir))

    print(f"[convert] Saving model to: {out_dir}")
    model.save_pretrained(str(out_dir))

    print("[convert] Done. You can now use this directory with from_pretrained / generate.sh.")


if __name__ == "__main__":
    main()
