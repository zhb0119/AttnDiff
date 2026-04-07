"""Similarity computation utilities."""

from typing import Optional

import numpy as np


def _sanitize_matrix(mat: np.ndarray, label: str) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    finite_mask = np.isfinite(mat)
    if not np.all(finite_mask):
        bad = int(np.size(mat) - np.count_nonzero(finite_mask))
        print(f"[Warning] Non-finite values detected in {label}: {bad} entries; replacing with 0.")
        mat = mat.copy()
        mat[~finite_mask] = 0.0
    return mat


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = _sanitize_matrix(X, "X")
    Y = _sanitize_matrix(Y, "Y")

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    X_norm = np.linalg.norm(X, ord="fro")
    Y_norm = np.linalg.norm(Y, ord="fro")

    if X_norm < 1e-12 or Y_norm < 1e-12:
        return 0.0

    X = X / X_norm
    Y = Y / Y_norm

    hsic = np.trace(X.T @ Y @ Y.T @ X)
    return float(hsic)


def fingerprint_to_matrix(
    vec: np.ndarray,
    L: int,
    H: int,
    N: int,
) -> np.ndarray:
    expected_len = L * H * N * N
    if vec.size != expected_len:
        raise ValueError(
            f"Fingerprint length mismatch: got {vec.size}, expected {expected_len}"
        )
    tensor = vec.reshape(L, H, N, N)
    mat = np.transpose(tensor, (2, 0, 1, 3)).reshape(N, -1)
    return mat


def compare_fingerprints(
    base_fingerprint: dict,
    target_fingerprint: dict,
    layer: Optional[int] = None,
) -> float:
    base_vec = np.array(base_fingerprint["fingerprint_vector"], dtype=np.float64)
    target_vec = np.array(target_fingerprint["fingerprint_vector"], dtype=np.float64)

    L = base_fingerprint["L"]
    H = base_fingerprint["H"]
    N = base_fingerprint["N"]

    if layer is not None:
        layer_idx = layer - 1
        base_mat = fingerprint_layer_to_matrix(base_vec, L, H, N, layer_idx)
        target_mat = fingerprint_layer_to_matrix(target_vec, L, H, N, layer_idx)
    else:
        base_mat = fingerprint_to_matrix(base_vec, L, H, N)
        target_mat = fingerprint_to_matrix(target_vec, L, H, N)

    return linear_cka(base_mat, target_mat)


def fingerprint_layer_to_matrix(
    vec: np.ndarray,
    L: int,
    H: int,
    N: int,
    layer_index: int,
) -> np.ndarray:
    if not (0 <= layer_index < L):
        raise ValueError(f"layer_index must be in [0, L-1], got {layer_index} for L={L}.")
    expected_len = L * H * N * N
    if vec.size != expected_len:
        raise ValueError(f"Fingerprint length mismatch: got {vec.size}, expected {expected_len}.")
    tensor = vec.reshape(L, H, N, N)
    layer_tensor = tensor[layer_index]
    mat = np.transpose(layer_tensor, (1, 0, 2)).reshape(N, -1)
    return mat
