#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on sys.path so that 'tool' can be imported
_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from tool.extract_attentions import load_model_and_tokenizer


@torch.no_grad()
def compute_mean_pooled_hidden_states(model, tokenizer, prompt: str, device: str):
    """Run a single prompt and return mean-pooled hidden states for each layer.

    Returns
    -------
    layer_vecs : np.ndarray
        Array of shape [L, D], where L is the number of transformer layers,
        and D is the hidden size. For each layer we mean-pool over the
        sequence dimension.
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(
        **encoded,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("Model did not return hidden_states.")

    # hidden_states[0] is usually embeddings; use 1..L as transformer layers
    hidden_layers = hidden_states[1:]
    layer_vecs = []
    for layer_h in hidden_layers:
        # layer_h: [batch, seq_len, hidden]
        h = layer_h[0]  # batch size 1
        vec = h.mean(dim=0)  # [hidden]
        layer_vecs.append(vec.cpu())

    layer_stack = torch.stack(layer_vecs, dim=0)  # [L, D]
    return layer_stack.numpy().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-layer hidden state activations for a dataset of prompts "
            "and save them as a single .npz file."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/dataset.json",
        help="Path to the dataset JSON file (default: dataset/dataset.json).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "HuggingFace model name or local path. This should match the model "
            "used to compute attentions / fingerprints."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output .npz path to store activations. If not set, defaults to "
            "output/activations/activations_<model_base>.npz."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device, e.g. 'cuda', 'cuda:0', or 'cpu'. If not set, "
            "automatically chooses CUDA if available, else CPU."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    model_name = args.model_name
    model_base = Path(model_name).name

    if args.out is None:
        out_path = Path("output/activations") / f"activations_{model_base}.npz"
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[HiddenExtract] Using device: {device}")
    print(f"[HiddenExtract] Loading model: {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_name, device)

    print(f"[HiddenExtract] Loading dataset from: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Dataset must be a non-empty list of items.")

    M = len(dataset)
    activations = None
    ids = []

    for idx, item in enumerate(tqdm(dataset, desc="Processing prompts")):
        sample_id = item.get("id", idx)
        prompt = item.get("original", None)
        if prompt is None:
            raise ValueError(
                f"Dataset item at index {idx} has no 'original' field: {item.keys()}"
            )

        layer_stack = compute_mean_pooled_hidden_states(
            model, tokenizer, prompt, device
        )  # [L, D]
        L, D = layer_stack.shape

        if activations is None:
            activations = np.zeros((M, L, D), dtype=np.float32)
            num_layers = L
            hidden_dim = D
            print(
                f"[HiddenExtract] Activation tensor shape will be: "
                f"(M={M}, L={num_layers}, D={hidden_dim})"
            )
        else:
            if layer_stack.shape != (num_layers, hidden_dim):
                raise ValueError(
                    "Hidden state shape mismatch: got "
                    f"{layer_stack.shape}, expected {(num_layers, hidden_dim)}."
                )

        activations[idx] = layer_stack
        ids.append(sample_id)

    if activations is None:
        raise RuntimeError("No activations extracted; dataset may be empty.")

    print(f"[HiddenExtract] Saving activations to: {out_path}")
    np.savez(
        out_path,
        activations=activations,
        num_samples=M,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        ids=np.asarray(ids, dtype=np.int64),
        model_base=model_base,
    )
    print("[HiddenExtract] Done.")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=3 python tool/extract_hidden_states.py \
  --model_name /home/kdz/data/OpenSourceModels/princeton-nlp/Sheared-LLaMA-1.3B-Pruned \
  --data_path dataset/dataset.json
"""