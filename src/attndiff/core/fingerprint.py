"""Fingerprint computation and loading utilities."""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def compute_fingerprint(
    original_data: list,
    corrupted_data: list,
    mode: str = "diff",
) -> dict:
    M = len(original_data)
    if len(corrupted_data) != M:
        raise ValueError(
            f"Mismatch in sample count: original={M}, corrupted={len(corrupted_data)}"
        )

    L = len(original_data[0])
    H = len(original_data[0][0])
    N = len(original_data[0][0][0])

    fingerprint_vector = []

    for layer_idx in tqdm(range(L), desc="Computing fingerprint"):
        for head_idx in range(H):
            diff_sum = np.zeros((N, N), dtype=np.float64)
            for sample_idx in range(M):
                orig_attn = np.array(
                    original_data[sample_idx][layer_idx][head_idx], dtype=np.float64
                )
                corr_attn = np.array(
                    corrupted_data[sample_idx][layer_idx][head_idx], dtype=np.float64
                )

                if mode == "diff":
                    diff = orig_attn - corr_attn
                elif mode == "orig":
                    diff = orig_attn
                elif mode == "base":
                    diff = corr_attn
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                diff_sum += diff

            avg_diff = diff_sum / M
            fingerprint_vector.append(avg_diff.flatten())

    fingerprint_vector = np.concatenate(fingerprint_vector)
    return {
        "fingerprint_vector": fingerprint_vector.tolist(),
        "L": L,
        "H": H,
        "N": N,
        "M": M,
        "mode": mode,
    }


def load_fingerprint(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
