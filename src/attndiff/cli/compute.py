#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
# Project root is 3 levels up from this file: src/attndiff/cli/compute.py -> AttnDiff-main/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_attention_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Attention file {path} is empty or not a list.")
    return data


def _infer_L_H_from_sample(sample_attention):
    L = len(sample_attention)
    if L == 0:
        raise ValueError("Attention has zero layers.")
    H = len(sample_attention[0])
    if H == 0:
        raise ValueError("Attention has zero heads in first layer.")
    return L, H


def _infer_model_name_from_paths(
    original_path_str: Optional[str],
    corrupted_path_str: Optional[str],
):
    candidates: list[str] = []
    if original_path_str is not None:
        candidates.append(original_path_str)
    if corrupted_path_str is not None:
        candidates.append(corrupted_path_str)

    for p_str in candidates:
        p = Path(p_str)
        stem = p.stem
        for suffix in ("_att_origin", "_att_perturb"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        # If no standard suffix matches, use stem as model name directly
        return stem

    return None


def compute_s_divergence_per_head(
    original_items,
    corrupted_items,
    pool_size: Optional[int] = None,
):
    if len(original_items) != len(corrupted_items):
        raise ValueError(
            f"Original and corrupted files have different lengths: "
            f"{len(original_items)} vs {len(corrupted_items)}"
        )

    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_scores = np.zeros((M, L, H), dtype=np.float64)
    if pool_size is not None:
        per_sample_grids = np.zeros((M, L, H, pool_size, pool_size), dtype=np.float64)
    else:
        per_sample_grids = None

    for idx, (orig, corr) in enumerate(
        tqdm(
            zip(original_items, corrupted_items),
            total=len(original_items),
            desc="Computing S_Divergence",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)
        attn_c = np.asarray(corr["attention"], dtype=np.float64)

        if attn_o.ndim != 4 or attn_c.ndim != 4:
            raise ValueError(
                "Attention tensors must be 4D [L, H, N, N], got "
                f"{attn_o.shape} and {attn_c.shape} at index {idx}."
            )

        # Ensure consistent L, H across samples and between original/corrupted
        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )
        if attn_c.shape[0] != L or attn_c.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H between original and corrupted at index "
                f"{idx}: original {attn_o.shape[:2]}, corrupted {attn_c.shape[:2]}"
            )

        # Allow different sequence lengths N: crop both to the common minimum N
        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N_c1, N_c2 = attn_c.shape[2], attn_c.shape[3]
        N = min(N_o1, N_o2, N_c1, N_c2)
        if N <= 0:
            raise ValueError(
                f"Invalid attention spatial shape at index {idx}: "
                f"original {attn_o.shape}, corrupted {attn_c.shape}"
            )
        attn_o = attn_o[:, :, :N, :N]
        attn_c = attn_c[:, :, :N, :N]

        diff = attn_c - attn_o
        frob = np.sqrt(np.sum(diff * diff, axis=(-1, -2)))

        scores += frob
        per_sample_scores[idx] = frob

        if pool_size is not None:
            for layer_idx in range(L):
                for head_idx in range(H):
                    grid = pool_diff_to_grid(diff[layer_idx, head_idx], out_size=pool_size)
                    per_sample_grids[idx, layer_idx, head_idx] = grid

    return scores, per_sample_scores, L, H, per_sample_grids


def _log_bucket_index_from_distance(d: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.zeros_like(d, dtype=np.int64)
    out = np.empty_like(d, dtype=np.int64)
    out[d == 0] = 0
    d_pos = d[d > 0]
    if d_pos.size > 0:
        idx = np.floor(np.log2(d_pos)).astype(np.int64) + 1
        idx = np.minimum(idx, k - 1)
        out[d > 0] = idx
    return out


def _get_tril_cache(
    N: int, k: int, cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
):
    key = (N, k)
    if key in cache:
        return cache[key]
    ii, jj = np.tril_indices(N)
    d = (ii - jj).astype(np.int64)
    b = _log_bucket_index_from_distance(d, k)
    cache[key] = (ii, jj, b)
    return cache[key]


def compute_log_bucket_mass_features_per_head(
    original_items,
    corrupted_items,
    bucket_k: int,
    normalize: bool = True,
):
    if len(original_items) != len(corrupted_items):
        raise ValueError(
            f"Original and corrupted files have different lengths: "
            f"{len(original_items)} vs {len(corrupted_items)}"
        )

    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    if bucket_k <= 0:
        raise ValueError(f"bucket_k must be positive, got {bucket_k}.")

    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_bucket = np.zeros((M, L, H, 2 * bucket_k), dtype=np.float64)
    tril_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for idx, (orig, corr) in enumerate(
        tqdm(
            zip(original_items, corrupted_items),
            total=len(original_items),
            desc="Computing Log-Bucket Features",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)
        attn_c = np.asarray(corr["attention"], dtype=np.float64)

        if attn_o.ndim != 4 or attn_c.ndim != 4:
            raise ValueError(
                "Attention tensors must be 4D [L, H, N, N], got "
                f"{attn_o.shape} and {attn_c.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )
        if attn_c.shape[0] != L or attn_c.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H between original and corrupted at index "
                f"{idx}: original {attn_o.shape[:2]}, corrupted {attn_c.shape[:2]}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N_c1, N_c2 = attn_c.shape[2], attn_c.shape[3]
        N = min(N_o1, N_o2, N_c1, N_c2)
        if N <= 0:
            raise ValueError(
                f"Invalid attention spatial shape at index {idx}: "
                f"original {attn_o.shape}, corrupted {attn_c.shape}"
            )
        attn_o = attn_o[:, :, :N, :N]
        attn_c = attn_c[:, :, :N, :N]

        diff = attn_c - attn_o
        frob = np.sqrt(np.sum(diff * diff, axis=(-1, -2)))
        scores += frob

        ii, jj, b = _get_tril_cache(N, bucket_k, tril_cache)
        for layer_idx in range(L):
            for head_idx in range(H):
                flat = diff[layer_idx, head_idx][ii, jj]
                pos = np.clip(flat, 0.0, None)
                neg = np.clip(-flat, 0.0, None)
                pos_bucket = np.bincount(b, weights=pos, minlength=bucket_k)
                neg_bucket = np.bincount(b, weights=neg, minlength=bucket_k)
                v = np.concatenate([pos_bucket, neg_bucket], axis=0)
                if normalize:
                    s = float(v.sum())
                    if s > 0:
                        v = v / s
                per_sample_bucket[idx, layer_idx, head_idx] = v

    return scores, per_sample_bucket, L, H


def compute_base_norm_per_head(original_items):
    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_scores = np.zeros((M, L, H), dtype=np.float64)

    for idx, orig in enumerate(
        tqdm(
            original_items,
            total=len(original_items),
            desc="Computing Base Norm",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)

        if attn_o.ndim != 4:
            raise ValueError(
                f"Attention tensors must be 4D [L, H, N, N], got {attn_o.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N = min(N_o1, N_o2)
        if N <= 0:
            raise ValueError(f"Invalid attention spatial shape at index {idx}: {attn_o.shape}")
        attn_o = attn_o[:, :, :N, :N]

        frob = np.sqrt(np.sum(attn_o * attn_o, axis=(-1, -2)))

        scores += frob
        per_sample_scores[idx] = frob

    return scores, per_sample_scores, L, H


def compute_topk_singular_values(mat: np.ndarray, k: int) -> np.ndarray:
    """Compute top-k singular values of a matrix.

    Args:
        mat: 2D numpy array of shape [N, N]
        k: number of top singular values to return

    Returns:
        1D numpy array of shape [k] containing top-k singular values.
        If matrix has fewer than k singular values, pad with zeros.
    """
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    result = np.zeros(k, dtype=np.float64)
    num_sv = min(len(S), k)
    result[:num_sv] = S[:num_sv]
    return result


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient measuring distribution inequality.
    0 = perfectly uniform, 1 = maximally concentrated.
    """
    x = np.abs(x).flatten()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    np.cumsum(x)
    return (2.0 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * x.sum()) / (n * x.sum())


def compute_ars_features(mat: np.ndarray, svd_k: int = 3) -> np.ndarray:
    """Compute Attention Response Signature (ARS) features for a single matrix.

    ARS is a fixed-dimension feature vector capturing structural properties:
    - Top-K singular values (energy distribution)
    - Spectral entropy (complexity)
    - Row/Column Gini coefficients (concentration)
    - Locality ratio (diagonal energy)
    - Diagonal dominance
    - Off-diagonal asymmetry
    - Quadrant energy ratios

    Args:
        mat: 2D numpy array of shape [N, N]
        svd_k: number of top singular values to include

    Returns:
        1D numpy array of shape [D] where D = svd_k + 10 (fixed dimension)
    """
    N = mat.shape[0]
    features = []

    # 1. SVD decomposition
    try:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        # Return zeros if SVD fails
        return np.zeros(svd_k + 10, dtype=np.float64)

    # 1a. Top-K singular values (normalized by Frobenius norm)
    f_norm = np.linalg.norm(mat, "fro")
    if f_norm > 0:
        sv_normalized = (
            S[:svd_k] / f_norm if len(S) >= svd_k else np.pad(S / f_norm, (0, svd_k - len(S)))
        )
    else:
        sv_normalized = np.zeros(svd_k)
    features.extend(sv_normalized.tolist())

    # 1b. Spectral entropy (normalized singular value distribution entropy)
    if f_norm > 0 and len(S) > 0:
        p = (S**2) / (f_norm**2)
        p = p[p > 1e-12]
        spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(S)) if len(S) > 1 else 0.0
    else:
        spectral_entropy = 0.0
    features.append(spectral_entropy)

    # 2. Row and Column Gini coefficients
    row_norms = np.linalg.norm(mat, axis=1)
    col_norms = np.linalg.norm(mat, axis=0)
    row_gini = gini_coefficient(row_norms)
    col_gini = gini_coefficient(col_norms)
    features.append(row_gini)
    features.append(col_gini)

    # 3. Locality ratio (energy within diagonal band, window=2)
    if f_norm > 0:
        window = min(2, N // 4) if N > 4 else 1
        mask = np.abs(np.arange(N)[:, None] - np.arange(N)) <= window
        local_energy = np.sum(mat[mask] ** 2)
        locality_ratio = local_energy / (f_norm**2)
    else:
        locality_ratio = 0.0
    features.append(locality_ratio)

    # 4. Diagonal dominance (diagonal energy / total energy)
    if f_norm > 0:
        diag_energy = np.sum(np.diag(mat) ** 2)
        diag_dominance = diag_energy / (f_norm**2)
    else:
        diag_dominance = 0.0
    features.append(diag_dominance)

    # 5. Off-diagonal asymmetry (upper vs lower triangle difference)
    upper = np.triu(mat, k=1)
    lower = np.tril(mat, k=-1)
    upper_energy = np.sum(upper**2)
    lower_energy = np.sum(lower**2)
    total_off_diag = upper_energy + lower_energy
    if total_off_diag > 0:
        asymmetry = (upper_energy - lower_energy) / total_off_diag
    else:
        asymmetry = 0.0
    features.append(asymmetry)

    # 6. Quadrant energy ratios (4 quadrants)
    mid = N // 2
    if mid > 0 and f_norm > 0:
        q1 = np.sum(mat[:mid, :mid] ** 2)  # top-left
        q2 = np.sum(mat[:mid, mid:] ** 2)  # top-right
        q3 = np.sum(mat[mid:, :mid] ** 2)  # bottom-left
        q4 = np.sum(mat[mid:, mid:] ** 2)  # bottom-right
        total = f_norm**2
        features.extend([q1 / total, q2 / total, q3 / total, q4 / total])
    else:
        features.extend([0.25, 0.25, 0.25, 0.25])

    return np.array(features, dtype=np.float64)


def compute_ars_features_per_head(
    original_items,
    corrupted_items,
    svd_k: int = 3,
):
    """Compute ARS (Attention Response Signature) features for each head's diff matrix.

    Args:
        original_items: list of original attention samples
        corrupted_items: list of corrupted attention samples
        svd_k: number of top singular values to include in ARS

    Returns:
        scores: [L, H] sum of Frobenius norms (for compatibility)
        per_sample_ars: [M, L, H, D] ARS features per sample per head (D = svd_k + 10)
        L: number of layers
        H: number of heads
        D: feature dimension
    """
    if len(original_items) != len(corrupted_items):
        raise ValueError(
            f"Original and corrupted files have different lengths: "
            f"{len(original_items)} vs {len(corrupted_items)}"
        )

    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    D = svd_k + 10  # Fixed feature dimension
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_ars = np.zeros((M, L, H, D), dtype=np.float64)

    for idx, (orig, corr) in enumerate(
        tqdm(
            zip(original_items, corrupted_items),
            total=len(original_items),
            desc="Computing ARS Features",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)
        attn_c = np.asarray(corr["attention"], dtype=np.float64)

        if attn_o.ndim != 4 or attn_c.ndim != 4:
            raise ValueError(
                "Attention tensors must be 4D [L, H, N, N], got "
                f"{attn_o.shape} and {attn_c.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )
        if attn_c.shape[0] != L or attn_c.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H between original and corrupted at index "
                f"{idx}: original {attn_o.shape[:2]}, corrupted {attn_c.shape[:2]}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N_c1, N_c2 = attn_c.shape[2], attn_c.shape[3]
        N = min(N_o1, N_o2, N_c1, N_c2)
        if N <= 0:
            raise ValueError(
                f"Invalid attention spatial shape at index {idx}: "
                f"original {attn_o.shape}, corrupted {attn_c.shape}"
            )
        attn_o = attn_o[:, :, :N, :N]
        attn_c = attn_c[:, :, :N, :N]

        diff = attn_c - attn_o
        frob = np.sqrt(np.sum(diff * diff, axis=(-1, -2)))
        scores += frob

        for layer_idx in range(L):
            for head_idx in range(H):
                ars = compute_ars_features(diff[layer_idx, head_idx], svd_k)
                per_sample_ars[idx, layer_idx, head_idx] = ars

    return scores, per_sample_ars, L, H, D


def compute_ars_features_base(
    original_items,
    svd_k: int = 3,
):
    """Compute ARS features for each head's original attention matrix.

    Args:
        original_items: list of original attention samples
        svd_k: number of top singular values to include in ARS

    Returns:
        scores: [L, H] sum of Frobenius norms (for compatibility)
        per_sample_ars: [M, L, H, D] ARS features per sample per head
        L: number of layers
        H: number of heads
        D: feature dimension
    """
    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    D = svd_k + 10
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_ars = np.zeros((M, L, H, D), dtype=np.float64)

    for idx, orig in enumerate(
        tqdm(
            original_items,
            total=len(original_items),
            desc="Computing ARS Features (Base)",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)

        if attn_o.ndim != 4:
            raise ValueError(
                f"Attention tensors must be 4D [L, H, N, N], got {attn_o.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N = min(N_o1, N_o2)
        if N <= 0:
            raise ValueError(f"Invalid attention spatial shape at index {idx}: {attn_o.shape}")
        attn_o = attn_o[:, :, :N, :N]

        frob = np.sqrt(np.sum(attn_o * attn_o, axis=(-1, -2)))
        scores += frob

        for layer_idx in range(L):
            for head_idx in range(H):
                ars = compute_ars_features(attn_o[layer_idx, head_idx], svd_k)
                per_sample_ars[idx, layer_idx, head_idx] = ars

    return scores, per_sample_ars, L, H, D


def pool_lower_triangular(mat: np.ndarray, out_size: int = 8) -> np.ndarray:
    """Pool only the lower triangular part of a matrix to a fixed size.

    Args:
        mat: 2D numpy array of shape [N, N] (lower triangular attention diff)
        out_size: output grid size P, resulting in P*(P+1)/2 features

    Returns:
        1D numpy array of shape [P*(P+1)/2] containing pooled lower triangular values
    """
    N = mat.shape[0]
    if N == 0:
        return np.zeros(out_size * (out_size + 1) // 2, dtype=np.float64)

    grid = np.zeros((out_size, out_size), dtype=np.float64)
    edges = np.linspace(0, N, out_size + 1, dtype=int)

    for i in range(out_size):
        r0, r1 = edges[i], edges[i + 1]
        if r1 <= r0:
            continue
        for j in range(i + 1):  # Only lower triangle (j <= i)
            c0, c1 = edges[j], edges[j + 1]
            if c1 <= c0:
                continue

            # Extract block
            block = mat[r0:r1, c0:c1]

            # For diagonal blocks, only use lower triangular part
            if i == j:
                block_mask = np.tril(np.ones_like(block, dtype=bool))
                if block_mask.sum() > 0:
                    grid[i, j] = np.mean(np.abs(block[block_mask]))
            else:
                # For off-diagonal blocks, use all elements
                if block.size > 0:
                    grid[i, j] = np.mean(np.abs(block))

    # Extract lower triangular elements as 1D feature vector
    mask = np.tril(np.ones((out_size, out_size), dtype=bool))
    return grid[mask]


def compute_pool_lower_features_per_head(
    original_items,
    corrupted_items,
    pool_size: int = 8,
):
    """Compute pooled lower triangular features for each head's diff matrix.

    Args:
        original_items: list of original attention samples
        corrupted_items: list of corrupted attention samples
        pool_size: output grid size P, feature dim = P*(P+1)/2

    Returns:
        scores: [L, H] sum of Frobenius norms (for compatibility)
        per_sample_pool: [M, L, H, D] pooled features per sample per head
        L: number of layers
        H: number of heads
        D: feature dimension = pool_size*(pool_size+1)/2
    """
    if len(original_items) != len(corrupted_items):
        raise ValueError(
            f"Original and corrupted files have different lengths: "
            f"{len(original_items)} vs {len(corrupted_items)}"
        )

    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    D = pool_size * (pool_size + 1) // 2  # Lower triangular elements
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_pool = np.zeros((M, L, H, D), dtype=np.float64)

    for idx, (orig, corr) in enumerate(
        tqdm(
            zip(original_items, corrupted_items),
            total=len(original_items),
            desc="Computing Pool Lower Features",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)
        attn_c = np.asarray(corr["attention"], dtype=np.float64)

        if attn_o.ndim != 4 or attn_c.ndim != 4:
            raise ValueError(
                "Attention tensors must be 4D [L, H, N, N], got "
                f"{attn_o.shape} and {attn_c.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )
        if attn_c.shape[0] != L or attn_c.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H between original and corrupted at index "
                f"{idx}: original {attn_o.shape[:2]}, corrupted {attn_c.shape[:2]}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N_c1, N_c2 = attn_c.shape[2], attn_c.shape[3]
        N = min(N_o1, N_o2, N_c1, N_c2)
        if N <= 0:
            raise ValueError(
                f"Invalid attention spatial shape at index {idx}: "
                f"original {attn_o.shape}, corrupted {attn_c.shape}"
            )
        attn_o = attn_o[:, :, :N, :N]
        attn_c = attn_c[:, :, :N, :N]

        diff = attn_c - attn_o
        frob = np.sqrt(np.sum(diff * diff, axis=(-1, -2)))
        scores += frob

        for layer_idx in range(L):
            for head_idx in range(H):
                pooled = pool_lower_triangular(diff[layer_idx, head_idx], pool_size)
                per_sample_pool[idx, layer_idx, head_idx] = pooled

    return scores, per_sample_pool, L, H, D


def compute_svd_features_per_head(
    original_items,
    corrupted_items,
    svd_k: int,
):
    """Compute SVD features (top-k singular values) for each head's diff matrix.

    Args:
        original_items: list of original attention samples
        corrupted_items: list of corrupted attention samples
        svd_k: number of top singular values to extract

    Returns:
        scores: [L, H] sum of Frobenius norms (for compatibility)
        per_sample_svd: [M, L, H, K] top-k singular values per sample per head
        L: number of layers
        H: number of heads
    """
    if len(original_items) != len(corrupted_items):
        raise ValueError(
            f"Original and corrupted files have different lengths: "
            f"{len(original_items)} vs {len(corrupted_items)}"
        )

    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_svd = np.zeros((M, L, H, svd_k), dtype=np.float64)

    for idx, (orig, corr) in enumerate(
        tqdm(
            zip(original_items, corrupted_items),
            total=len(original_items),
            desc="Computing SVD Features",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)
        attn_c = np.asarray(corr["attention"], dtype=np.float64)

        if attn_o.ndim != 4 or attn_c.ndim != 4:
            raise ValueError(
                "Attention tensors must be 4D [L, H, N, N], got "
                f"{attn_o.shape} and {attn_c.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )
        if attn_c.shape[0] != L or attn_c.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H between original and corrupted at index "
                f"{idx}: original {attn_o.shape[:2]}, corrupted {attn_c.shape[:2]}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N_c1, N_c2 = attn_c.shape[2], attn_c.shape[3]
        N = min(N_o1, N_o2, N_c1, N_c2)
        if N <= 0:
            raise ValueError(
                f"Invalid attention spatial shape at index {idx}: "
                f"original {attn_o.shape}, corrupted {attn_c.shape}"
            )
        attn_o = attn_o[:, :, :N, :N]
        attn_c = attn_c[:, :, :N, :N]

        diff = attn_c - attn_o
        frob = np.sqrt(np.sum(diff * diff, axis=(-1, -2)))
        scores += frob

        for layer_idx in range(L):
            for head_idx in range(H):
                sv = compute_topk_singular_values(diff[layer_idx, head_idx], svd_k)
                per_sample_svd[idx, layer_idx, head_idx] = sv

    return scores, per_sample_svd, L, H


def compute_svd_features_base(
    original_items,
    svd_k: int,
):
    """Compute SVD features (top-k singular values) for each head's original attention.

    Args:
        original_items: list of original attention samples
        svd_k: number of top singular values to extract

    Returns:
        scores: [L, H] sum of Frobenius norms (for compatibility)
        per_sample_svd: [M, L, H, K] top-k singular values per sample per head
        L: number of layers
        H: number of heads
    """
    L, H = _infer_L_H_from_sample(original_items[0]["attention"])
    M = len(original_items)
    scores = np.zeros((L, H), dtype=np.float64)
    per_sample_svd = np.zeros((M, L, H, svd_k), dtype=np.float64)

    for idx, orig in enumerate(
        tqdm(
            original_items,
            total=len(original_items),
            desc="Computing SVD Features (Base)",
        )
    ):
        attn_o = np.asarray(orig["attention"], dtype=np.float64)

        if attn_o.ndim != 4:
            raise ValueError(
                f"Attention tensors must be 4D [L, H, N, N], got {attn_o.shape} at index {idx}."
            )

        if attn_o.shape[0] != L or attn_o.shape[1] != H:
            raise ValueError(
                "Inconsistent L/H across samples: expected (L, H) = "
                f"({L}, {H}), got ({attn_o.shape[0]}, {attn_o.shape[1]}) at index {idx}"
            )

        N_o1, N_o2 = attn_o.shape[2], attn_o.shape[3]
        N = min(N_o1, N_o2)
        if N <= 0:
            raise ValueError(f"Invalid attention spatial shape at index {idx}: {attn_o.shape}")
        attn_o = attn_o[:, :, :N, :N]

        frob = np.sqrt(np.sum(attn_o * attn_o, axis=(-1, -2)))
        scores += frob

        for layer_idx in range(L):
            for head_idx in range(H):
                sv = compute_topk_singular_values(attn_o[layer_idx, head_idx], svd_k)
                per_sample_svd[idx, layer_idx, head_idx] = sv

    return scores, per_sample_svd, L, H


def compute_diff_for_head(
    original_items,
    corrupted_items,
    layer_idx: int,
    head_idx: int,
):
    diff_sum = None
    N_ref = None
    count = 0

    for idx, (orig, corr) in enumerate(zip(original_items, corrupted_items)):
        attn_o = np.asarray(orig["attention"][layer_idx][head_idx], dtype=np.float64)
        attn_c = np.asarray(corr["attention"][layer_idx][head_idx], dtype=np.float64)

        # Allow different spatial sizes by cropping both to common minimum N
        if attn_o.ndim != 2 or attn_c.ndim != 2:
            raise ValueError(
                "Per-head attention matrices must be 2D, got "
                f"{attn_o.shape} and {attn_c.shape} for layer {layer_idx}, "
                f"head {head_idx} at sample {idx}."
            )
        N = min(attn_o.shape[0], attn_o.shape[1], attn_c.shape[0], attn_c.shape[1])
        if N <= 0:
            raise ValueError(
                f"Invalid per-head attention shape for layer {layer_idx}, head {head_idx} "
                f"at sample {idx}: {attn_o.shape} vs {attn_c.shape}"
            )
        attn_o = attn_o[:N, :N]
        attn_c = attn_c[:N, :N]

        if diff_sum is None:
            N_ref = N
            diff_sum = attn_c - attn_o
        else:
            if N < N_ref:
                diff_sum = diff_sum[:N, :N]
                N_ref = N
                diff = attn_c - attn_o
            else:
                diff = (attn_c - attn_o)[:N_ref, :N_ref]
            diff_sum += diff
        count += 1

    if count == 0:
        raise ValueError("No samples found to compute attention difference.")

    diff_mean = diff_sum / count
    return diff_mean, N_ref


def pool_diff_to_grid(diff_mat, out_size: int = 10):
    N = diff_mat.shape[0]
    if diff_mat.shape[0] != diff_mat.shape[1]:
        raise ValueError("diff_mat must be square for pooling.")

    grid = np.zeros((out_size, out_size), dtype=np.float64)
    if N == 0:
        return grid

    edges = np.linspace(0, N, out_size + 1, dtype=int)
    for i in range(out_size):
        r0, r1 = edges[i], edges[i + 1]
        if r1 <= r0:
            continue
        for j in range(out_size):
            c0, c1 = edges[j], edges[j + 1]
            if c1 <= c0:
                continue
            block = diff_mat[r0:r1, c0:c1]
            if block.size > 0:
                grid[i, j] = float(np.mean(np.abs(block)))

    return grid


def ensure_attention_files(
    original_path: Path,
    corrupted_path: Optional[Path],
    model_name: Optional[str],
    attn_device: Optional[str],
):
    # If both original and corrupted are needed
    if corrupted_path is not None:
        if original_path.exists() and corrupted_path.exists():
            return
    else:
        # Only need original
        if original_path.exists():
            return

    if model_name is None:
        raise FileNotFoundError(
            "Attention files not found and model_name is None; cannot auto-generate "
            "attentions. Please provide --model_name or manually generate the "
            "attention JSON files."
        )

    data_path = _resolve_path(Path("dataset/dataset.json"))
    if not data_path.exists():
        if corrupted_path is not None:
            raise FileNotFoundError(
                f"Attention files {original_path} / {corrupted_path} not found, and "
                f"dataset {data_path} is missing. Please ensure the dataset exists or "
                "adjust compute_W.py to point to the correct dataset path."
            )
        else:
            raise FileNotFoundError(
                f"Attention file {original_path} not found, and dataset {data_path} "
                "is missing. Please ensure the dataset exists or adjust compute_W.py "
                "to point to the correct dataset path."
            )

    print(
        "Attention files not found; automatically extracting attentions for "
        f"model {model_name} using dataset {data_path} ..."
    )
    original_path.parent.mkdir(parents=True, exist_ok=True)
    if corrupted_path is not None:
        corrupted_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from attndiff.utils.extract_attentions import process_dataset
    except ImportError as e:
        raise ImportError(
            "Failed to import process_dataset from extract_attentions while trying to "
            "auto-generate attention files. Please ensure that transformers and "
            "huggingface-hub versions are compatible, or pre-generate the attention "
            f"files manually. Original error: {e}"
        ) from e

    process_dataset(
        data_path=data_path,
        model_name=model_name,
        out_original=original_path,
        out_corrupted=corrupted_path,
        device=attn_device,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: compute per-head attention statistics/differences "
            "and construct high-dimensional fingerprints for those heads."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Optional model name or path. When provided and --original/--corrupted/--out "
            "are not set, file paths will be derived from the final component of "
            "model_name."
        ),
    )
    parser.add_argument(
        "--original",
        type=str,
        default=None,
        help=(
            "Path to attentions_original.json. If not set but --model_name is given, "
            "defaults to output/attention/<model>_att_origin.json."
        ),
    )
    parser.add_argument(
        "--corrupted",
        type=str,
        default=None,
        help=(
            "Path to attentions_corrupted.json. If not set but --model_name is given, "
            "defaults to output/attention/<model>_att_perturb.json."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of heads to select per layer (top-k by divergence)",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=None,
        help="Optional pooling size. If set, fingerprints use pooled_diff (N=pool_size). If not set, fingerprints use raw diff (original N).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["diff", "base", "orig"],
        default="diff",
        help=(
            "Fingerprint mode: 'diff' uses attention differences between corrupted "
            "and original prompts over the dataset; 'orig' uses only the original "
            "attentions over all dataset items (per-sample head norms); 'base' "
            "uses only the base/original attention from the dataset item with id "
            "== 3 as the fingerprint."
        ),
    )
    parser.add_argument(
        "--base_id",
        type=int,
        default=3,
        help=(
            "Sample id to use in 'base' mode. This id is looked up in the original "
            "attention JSON (default: 3)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output JSON file storing scores, selected heads, and fingerprint vector. "
            "If not set but --model_name is given, defaults to "
            "output/comput_W/fingerprint_<model>.json."
        ),
    )
    parser.add_argument(
        "--attn_device",
        type=str,
        default=None,
        help=(
            "Optional device for auto-generating attention files, e.g. 'cuda:5', "
            "'cuda:6', or 'cpu'. If not set, extract_attentions will pick a device "
            "automatically."
        ),
    )
    parser.add_argument(
        "--svd_k",
        type=int,
        default=None,
        help=(
            "If set, extract top-k singular values from each head's N×N matrix "
            "instead of computing a single Frobenius norm scalar. This preserves "
            "more structural information. Output will contain 'per_sample_svd' "
            "with shape [M, L, H, K]. Recommended values: 5-10."
        ),
    )
    parser.add_argument(
        "--ars",
        action="store_true",
        help=(
            "If set, compute Attention Response Signature (ARS) features instead "
            "of simple SVD or Frobenius norm. ARS is a fixed-dimension feature "
            "vector (D = svd_k + 10, default svd_k=3) capturing: top-K singular "
            "values, spectral entropy, row/col Gini, locality ratio, diagonal "
            "dominance, off-diagonal asymmetry, and quadrant energy ratios. "
            "Output will contain 'per_sample_ars' with shape [M, L, H, D]."
        ),
    )
    parser.add_argument(
        "--ars_svd_k",
        type=int,
        default=3,
        help=(
            "Number of top singular values to include in ARS features. "
            "Total ARS dimension = ars_svd_k + 10. Default: 3."
        ),
    )
    parser.add_argument(
        "--pool_lower",
        type=int,
        default=None,
        help=(
            "If set, pool only the lower triangular part of each head's N×N "
            "diff matrix to a P×P grid, then extract lower triangular elements. "
            "This avoids the upper triangular zeros (causal mask) that would "
            "introduce false similarity. Feature dimension = P*(P+1)/2. "
            "Recommended values: 8-12."
        ),
    )
    parser.add_argument(
        "--log_bucket_k",
        type=int,
        default=None,
        help=(
            "If set (diff mode only), compute log-distance bucket features from each "
            "head's lower-triangular attention difference ΔA (attn_probs). Each head "
            "is represented as a 2K vector [pos_mass_by_bucket, neg_mass_by_bucket] "
            "with shape [M, L, H, 2K], stored in 'per_sample_log_bucket'."
        ),
    )
    parser.add_argument(
        "--log_bucket_normalize",
        action="store_true",
        help=(
            "If set (with --log_bucket_k), normalize each per-head 2K vector by its "
            "total mass so it captures distribution shape rather than total energy."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name

    # Derive model base name from model_name if provided
    model_base = Path(model_name).name if model_name is not None else None

    inferred_model_name = None
    if model_name is None:
        inferred_model_name = _infer_model_name_from_paths(
            args.original,
            args.corrupted,
        )
    display_model_name = model_name if model_name is not None else inferred_model_name

    print("\n[ComputeW] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[ComputeW] 🧩  Fingerprint Configuration")
    print("[ComputeW] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(
        f"[ComputeW] 📦 Model name      : "
        f"{display_model_name if display_model_name is not None else 'Unknown'}"
    )
    print(f"[ComputeW] 🎯 Mode            : {args.mode}")
    if args.pool_lower is not None:
        pool_dim = args.pool_lower * (args.pool_lower + 1) // 2
        print(f"[ComputeW] 📐 Pool Lower      : P={args.pool_lower} (D={pool_dim})")
    elif args.ars:
        print(
            f"[ComputeW] 🧬 ARS enabled     : True (D = {args.ars_svd_k} + 10 = {args.ars_svd_k + 10})"
        )
    elif args.svd_k is not None:
        print(f"[ComputeW] 🔢 SVD top-k       : {args.svd_k}")
    if args.attn_device is not None:
        print(f"[ComputeW] 🖥️  Attn device     : {args.attn_device}")
    else:
        print("[ComputeW] 🖥️  Attn device     : auto")
    print("[ComputeW] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Resolve original attention path (always needed)
    if args.original is not None:
        original_path = _resolve_path(Path(args.original))
    else:
        if model_base is None:
            raise ValueError(
                "Either --original must be provided or --model_name must be set "
                "to derive the default original attention path."
            )
        original_path = _resolve_path(Path("output/attention") / f"{model_base}_att_origin.json")

    # Resolve corrupted attention path only when needed (diff mode)
    if args.mode == "diff":
        if args.corrupted is not None:
            corrupted_path: Optional[Path] = _resolve_path(Path(args.corrupted))
        else:
            if model_base is None:
                raise ValueError(
                    "Either --corrupted must be provided or --model_name must be set "
                    "to derive the default corrupted attention path."
                )
            corrupted_path = _resolve_path(
                Path("output/attention") / f"{model_base}_att_perturb.json"
            )
    else:
        corrupted_path = None

    # Auto-generate attention files if they are missing
    if args.mode == "diff":
        ensure_attention_files(original_path, corrupted_path, model_name, args.attn_device)
    else:
        # In base mode, only ensure original exists
        ensure_attention_files(original_path, None, model_name, args.attn_device)

    # Resolve output fingerprint path
    if args.out is not None:
        out_path = _resolve_path(Path(args.out))
    else:
        if model_base is None:
            raise ValueError(
                "Either --out must be provided or --model_name must be set "
                "to derive the default fingerprint path."
            )
        if args.mode == "orig":
            out_path = _resolve_path(
                Path("output/base_fingerprint") / f"fingerprint_{model_base}.json"
            )
        else:
            out_path = _resolve_path(Path("output/comput_W") / f"fingerprint_{model_base}.json")

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ComputeW] Loading original attentions from: {original_path}")
    original_items = load_attention_file(original_path)

    if args.mode == "diff":
        if corrupted_path is None:
            raise ValueError("corrupted_path must not be None in diff mode.")
        print(f"[ComputeW] Loading corrupted attentions from: {corrupted_path}")
        corrupted_items = load_attention_file(corrupted_path)
    else:
        corrupted_items = None

    # Build per-layer/head data used for fingerprint construction
    layers_out = []
    per_sample_grids = None
    per_sample_svd = None  # For SVD features
    per_sample_ars = None  # For ARS features
    per_sample_pool_lower = None  # For pool lower triangular features
    per_sample_log_bucket = None  # For log-distance bucket features
    ars_dim = None
    pool_lower_dim = None
    log_bucket_k = None

    if args.mode == "diff":
        # Diff mode: need S_Divergence scores based on original/corrupted pairs
        if args.pool_lower is not None:
            # Use pool lower triangular features
            print(
                f"\n[ComputeW] 🔍 [Mode: diff + Pool Lower] Pooling lower triangular to {args.pool_lower}x{args.pool_lower} grid ..."
            )
            scores, per_sample_pool_lower, L, H, pool_lower_dim = (
                compute_pool_lower_features_per_head(
                    original_items,
                    corrupted_items,
                    pool_size=args.pool_lower,
                )
            )
            print(
                "[ComputeW] ┌──────────────── Diff mode (Pool Lower) dimensions ────────────────┐"
            )
            print(f"[ComputeW] │ Score matrix [L, H]       : {scores.shape} (L={L}, H={H})")
            print(
                f"[ComputeW] │ Pool features [M, L, H, D]: {per_sample_pool_lower.shape} (D={pool_lower_dim})"
            )
            print("[ComputeW] │ Diff source               : corrupted - original attentions")
            print(
                "[ComputeW] └─────────────────────────────────────────────────────────────────────┘"
            )
            # For compatibility, compute per_sample_scores as sum of pooled values
            per_sample_scores = np.sum(per_sample_pool_lower, axis=-1)  # [M, L, H]
        elif args.ars:
            # Use ARS features
            print(
                "\n[ComputeW] 🔍 [Mode: diff + ARS] Computing Attention Response Signature for each head ..."
            )
            scores, per_sample_ars, L, H, ars_dim = compute_ars_features_per_head(
                original_items,
                corrupted_items,
                svd_k=args.ars_svd_k,
            )
            print("[ComputeW] ┌──────────────── Diff mode (ARS) dimensions ────────────────┐")
            print(f"[ComputeW] │ Score matrix [L, H]       : {scores.shape} (L={L}, H={H})")
            print(f"[ComputeW] │ ARS features [M, L, H, D] : {per_sample_ars.shape} (D={ars_dim})")
            print("[ComputeW] │ Diff source               : corrupted - original attentions")
            print("[ComputeW] └────────────────────────────────────────────────────────────┘")
            # For compatibility, compute per_sample_scores from ARS singular-value slice
            per_sample_scores = np.sqrt(
                np.sum(per_sample_ars[:, :, :, : args.ars_svd_k] ** 2, axis=-1)
            )
        elif args.svd_k is not None:
            # Use SVD features instead of scalar norms
            print(
                f"\n[ComputeW] 🔍 [Mode: diff + SVD] Computing top-{args.svd_k} singular values for each head ..."
            )
            scores, per_sample_svd, L, H = compute_svd_features_per_head(
                original_items,
                corrupted_items,
                svd_k=args.svd_k,
            )
            print("[ComputeW] ┌──────────────── Diff mode (SVD) dimensions ────────────────┐")
            print(f"[ComputeW] │ Score matrix [L, H]       : {scores.shape} (L={L}, H={H})")
            print(f"[ComputeW] │ SVD features [M, L, H, K] : {per_sample_svd.shape}")
            # For compatibility, also compute per_sample_scores as Frobenius norms
            per_sample_scores = np.sqrt(np.sum(per_sample_svd**2, axis=-1))  # [M, L, H]
        else:
            print(
                "\n[ComputeW] 🔍 [Mode: diff] Computing per-sample head norms "
                "from attention differences ..."
            )
            scores, per_sample_scores, L, H, per_sample_grids = compute_s_divergence_per_head(
                original_items,
                corrupted_items,
                pool_size=args.pool_size,
            )
            print("[ComputeW] ┌──────────────── Diff mode dimensions ────────────────┐")
            print(f"[ComputeW] │ Score matrix [L, H] : {scores.shape} (L={L}, H={H})")
            print("[ComputeW] │ Diff source         : corrupted - original attentions")
            print("[ComputeW] └──────────────────────────────────────────────────────┘")

        if args.log_bucket_k is not None:
            log_bucket_k = int(args.log_bucket_k)
            print(
                f"\n[ComputeW] 🔍 [Mode: diff + Log-Bucket] Computing log-distance bucket features "
                f"with K={log_bucket_k} ..."
            )
            _, per_sample_log_bucket, _, _ = compute_log_bucket_mass_features_per_head(
                original_items,
                corrupted_items,
                bucket_k=log_bucket_k,
                normalize=bool(args.log_bucket_normalize),
            )

        # Diff mode: use attention differences between corrupted and original
        print("[ComputeW] ⚙️  Building fingerprint from attention differences ...")

        for layer_idx in range(L):
            heads_out = []
            for head_idx in range(H):
                score = float(scores[layer_idx, head_idx])
                diff_mat, N = compute_diff_for_head(
                    original_items,
                    corrupted_items,
                    layer_idx,
                    head_idx,
                )

                head_data = {
                    "head": int(head_idx),
                    "score": float(score),
                    "N": int(N),
                    "diff": diff_mat.tolist(),  # [N][N]
                }

                if args.pool_size is not None:
                    pooled = pool_diff_to_grid(diff_mat, out_size=args.pool_size)
                    head_data["pooled_diff"] = pooled.tolist()

                heads_out.append(head_data)
            layers_out.append({"layer": int(layer_idx), "heads": heads_out})
    elif args.mode == "orig":
        if args.ars:
            # Use ARS features
            print(
                "\n[ComputeW] 🔍 [Mode: orig + ARS] Computing Attention Response Signature for each head ..."
            )
            scores, per_sample_ars, L, H, ars_dim = compute_ars_features_base(
                original_items,
                svd_k=args.ars_svd_k,
            )
            print("[ComputeW] ┌──────────────── Orig mode (ARS) dimensions ────────────────┐")
            print(f"[ComputeW] │ Score matrix [L, H]       : {scores.shape} (L={L}, H={H})")
            print(f"[ComputeW] │ ARS features [M, L, H, D] : {per_sample_ars.shape} (D={ars_dim})")
            print("[ComputeW] │ Source                    : original attentions only")
            print("[ComputeW] └────────────────────────────────────────────────────────────┘")
            # For compatibility, compute per_sample_scores from first svd_k features
            per_sample_scores = np.sqrt(
                np.sum(per_sample_ars[:, :, :, : args.ars_svd_k] ** 2, axis=-1)
            )
        elif args.svd_k is not None:
            # Use SVD features instead of scalar norms
            print(
                f"\n[ComputeW] 🔍 [Mode: orig + SVD] Computing top-{args.svd_k} singular values "
                "from original attentions ..."
            )
            scores, per_sample_svd, L, H = compute_svd_features_base(
                original_items,
                svd_k=args.svd_k,
            )
            print("[ComputeW] ┌──────────────── Orig mode (SVD) dimensions ────────────────┐")
            print(f"[ComputeW] │ Score matrix [L, H]       : {scores.shape} (L={L}, H={H})")
            print(f"[ComputeW] │ SVD features [M, L, H, K] : {per_sample_svd.shape}")
            print("[ComputeW] │ Source                    : original attentions only")
            print("[ComputeW] └────────────────────────────────────────────────────────────┘")
            # For compatibility, also compute per_sample_scores as Frobenius norms
            per_sample_scores = np.sqrt(np.sum(per_sample_svd**2, axis=-1))  # [M, L, H]
        else:
            print(
                "\n[ComputeW] 🔍 [Mode: orig] Computing per-sample head norms "
                "from original attentions ..."
            )
            scores, per_sample_scores, L, H = compute_base_norm_per_head(original_items)
            print("[ComputeW] ┌──────────────── Orig mode dimensions ────────────────┐")
            print(f"[ComputeW] │ Score matrix [L, H] : {scores.shape} (L={L}, H={H})")
            print("[ComputeW] │ Source              : original attentions only")
            print("[ComputeW] └──────────────────────────────────────────────────────┘")

        for layer_idx in range(L):
            heads_out = []
            for head_idx in range(H):
                score = float(scores[layer_idx, head_idx])

                head_data = {
                    "head": int(head_idx),
                    "score": float(score),
                }

                heads_out.append(head_data)
            layers_out.append({"layer": int(layer_idx), "heads": heads_out})
    else:
        # Base mode: use only original attentions from a specific dataset item
        print(
            f"\n[ComputeW] 📘 [Mode: base] Using original attentions from dataset item id == {args.base_id}."
        )

        base_items = [item for item in original_items if item.get("id") == args.base_id]
        if not base_items:
            raise ValueError(
                "Mode 'base' selected, but no item with id == "
                f"{args.base_id} found in {original_path}."
            )
        base_item = base_items[0]

        attn_base = np.asarray(base_item["attention"], dtype=np.float64)
        if attn_base.ndim != 4:
            raise ValueError(
                f"Base attention for id={args.base_id} must have 4 dims [L,H,N,N], got {attn_base.shape}."
            )
        # Infer L,H from this base sample and create dummy scores (not used in base mode)
        L, H = attn_base.shape[0], attn_base.shape[1]
        scores = np.zeros((L, H), dtype=np.float64)

        N_base = attn_base.shape[2]
        print("[ComputeW] ┌──────────────── Base mode dimensions ────────────────┐")
        print(
            f"[ComputeW] │ Attention tensor [L, H, N, N] : {attn_base.shape} (L={L}, H={H}, N={N_base})"
        )
        print("[ComputeW] │ Base source                   : single original attention sample")
        print("[ComputeW] └──────────────────────────────────────────────────────┘")

        for layer_idx in range(L):
            heads_out = []
            for head_idx in range(H):
                score = float(scores[layer_idx, head_idx])
                mat = np.asarray(attn_base[layer_idx][head_idx], dtype=np.float64)
                if mat.shape[0] != mat.shape[1]:
                    raise ValueError(
                        "Base attention matrix is not square for layer "
                        f"{layer_idx}, head {head_idx}: {mat.shape}"
                    )
                N = mat.shape[0]

                head_data = {
                    "head": int(head_idx),
                    "score": float(score),
                    "N": int(N),
                    # Reuse 'diff' field to store base attention matrix
                    "diff": mat.tolist(),  # [N][N]
                }

                if args.pool_size is not None:
                    pooled = pool_diff_to_grid(mat, out_size=args.pool_size)
                    head_data["pooled_diff"] = pooled.tolist()

                heads_out.append(head_data)
            layers_out.append({"layer": int(layer_idx), "heads": heads_out})

    # Summary of attention difference matrices
    total_heads = sum(len(layer["heads"]) for layer in layers_out)
    if args.mode == "orig":
        Ns = []
        shapes_str = "N/A"
    else:
        if total_heads > 0:
            Ns_set = {head["N"] for layer in layers_out for head in layer["heads"]}
            Ns = sorted(Ns_set)
            shapes_str = ", ".join(f"{N}x{N}" for N in Ns)
        else:
            Ns = []
            shapes_str = "N/A"
    print("\n[ComputeW] 📊 Attention summary")
    print("[ComputeW] ────────────────────────────────────────")
    print(f"[ComputeW] • Total selected heads          : {total_heads}")
    print(f"[ComputeW] • Unique attention matrix sizes : {shapes_str}")

    # Construct high-dimensional fingerprint by concatenating flattened features
    if args.mode in ("diff", "orig"):
        M = len(original_items)
        if M == 0:
            N_global = 0
            K = 0
            D_flat = 0
            fingerprint_vec = []
            print("[ComputeW] No samples found; fingerprint vector is empty.")
        else:
            K = M
            D_flat = L * H
            F_mat = per_sample_scores.reshape(M, D_flat)

            print("\n[ComputeW] 📈 Fingerprint summary")
            print("[ComputeW] ────────────────────────────────────────")
            print(f"[ComputeW] • Fingerprint matrix [M, L*H] : {F_mat.shape} (M={M}, L*H={D_flat})")
            print(f"[ComputeW] • Fingerprint vector length    : {M * D_flat}")

            fingerprint_vec = F_mat.reshape(-1).tolist()
            N_global = M
    else:
        if total_heads > 0:
            K = total_heads

            if args.pool_size is not None:
                final_N = args.pool_size
                print(
                    f"[ComputeW] 🧮 Using pooled grid size N={final_N} for fingerprint construction ..."
                )
            else:
                # Use raw diffs
                # Ensure N is consistent or crop to min
                if not Ns:
                    final_N = 0
                else:
                    final_N = min(Ns)
                print(
                    f"[ComputeW] 🧮 Using raw diff size N={final_N} for fingerprint construction ..."
                )

            D_flat = final_N * final_N
            F_mat = np.zeros((K, D_flat), dtype=np.float64)

            row_idx = 0
            for layer in layers_out:
                for head in layer["heads"]:
                    if args.pool_size is not None:
                        if "pooled_diff" not in head:
                            raise ValueError(f"pooled_diff missing for head {head['head']}")
                        mat = np.asarray(head["pooled_diff"], dtype=np.float64)
                    else:
                        mat = np.asarray(head["diff"], dtype=np.float64)

                    # Check/Crop shape
                    if mat.shape[0] != final_N or mat.shape[1] != final_N:
                        if mat.shape[0] >= final_N and mat.shape[1] >= final_N:
                            mat = mat[:final_N, :final_N]
                        else:
                            raise ValueError(
                                f"Matrix shape {mat.shape} smaller than target N={final_N}"
                            )

                    F_mat[row_idx] = mat.reshape(-1)
                    row_idx += 1

            print("\n[ComputeW] 📈 Fingerprint summary")
            print("[ComputeW] ────────────────────────────────────────")
            print(f"[ComputeW] • Fingerprint matrix [K, N^2] : {F_mat.shape} (K={K}, N^2={D_flat})")
            print(f"[ComputeW] • Fingerprint vector length    : {K * D_flat}")

            fingerprint_vec = F_mat.reshape(-1).tolist()
            N_global = final_N
        else:
            N_global = 0
            K = 0
            D_flat = 0
            fingerprint_vec = []
            print("[ComputeW] No attention diff matrices found; fingerprint vector is empty.")

    out_obj = {
        "L": int(L),
        "H": int(H),
        "k_per_layer": int(H),  # Using all heads
        "scores": scores.tolist(),  # [L][H]
        "trajectory_heads": layers_out,
        "N_global": int(N_global),
        "fingerprint_vector_length": int(len(fingerprint_vec)),
        "fingerprint_vector": fingerprint_vec,
    }

    if args.mode == "diff" and per_sample_grids is not None:
        out_obj["per_sample_grids"] = per_sample_grids.tolist()

    if args.mode in ("diff", "orig"):
        out_obj["num_samples"] = int(len(original_items))
        out_obj["per_sample_scores"] = per_sample_scores.tolist()

        if per_sample_log_bucket is not None:
            out_obj["log_bucket_k"] = int(
                log_bucket_k if log_bucket_k is not None else per_sample_log_bucket.shape[-1] // 2
            )
            out_obj["log_bucket_normalize"] = bool(args.log_bucket_normalize)
            out_obj["per_sample_log_bucket"] = per_sample_log_bucket.tolist()  # [M, L, H, 2K]

        # Save SVD features if computed
        if per_sample_svd is not None:
            out_obj["svd_k"] = int(args.svd_k)
            out_obj["per_sample_svd"] = per_sample_svd.tolist()  # [M, L, H, K]

        # Save ARS features if computed
        if per_sample_ars is not None:
            out_obj["ars_svd_k"] = int(args.ars_svd_k)
            out_obj["ars_dim"] = int(ars_dim)
            out_obj["per_sample_ars"] = per_sample_ars.tolist()  # [M, L, H, D]

        # Save pool lower features if computed
        if per_sample_pool_lower is not None:
            out_obj["pool_lower_size"] = int(args.pool_lower)
            out_obj["pool_lower_dim"] = int(pool_lower_dim)
            out_obj["per_sample_pool_lower"] = per_sample_pool_lower.tolist()  # [M, L, H, D]

    print(f"[ComputeW] Saving attention diff matrices and fingerprint vector to: {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False)
    print("[ComputeW] Done.")


if __name__ == "__main__":
    main()
