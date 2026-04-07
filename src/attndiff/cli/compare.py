#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA


def _sanitize_matrix(mat: np.ndarray, label: str) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    finite_mask = np.isfinite(mat)
    if not np.all(finite_mask):
        bad = int(np.size(mat) - np.count_nonzero(finite_mask))
        print(f"[Warning] Non-finite values detected in {label}: {bad} entries; replacing with 0.")
        mat = mat.copy()
        mat[~finite_mask] = 0.0
    return mat


def fingerprint_to_matrix(
    vec: np.ndarray,
    L: int,
    H: int,
    N: int,
) -> np.ndarray:
    expected_len = L * H * N * N
    if vec.size != expected_len:
        raise ValueError(
            "Fingerprint length mismatch: got "
            f"{vec.size}, expected {expected_len} for L={L}, H={H}, N={N}."
        )
    tensor = vec.reshape(L, H, N, N)
    mat = np.transpose(tensor, (2, 0, 1, 3)).reshape(N, -1)
    return mat


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
        raise ValueError(
            "Fingerprint length mismatch: got "
            f"{vec.size}, expected {expected_len} for L={L}, H={H}, N={N}."
        )
    tensor = vec.reshape(L, H, N, N)
    layer_tensor = tensor[layer_index : layer_index + 1, :, :, :]
    mat = np.transpose(layer_tensor, (2, 0, 1, 3)).reshape(N, -1)
    return mat


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA requires the same number of rows, got {X.shape[0]} and {Y.shape[0]}."
        )
    # Gram matrices in sample space (N x N)
    K = X @ X.T
    L = Y @ Y.T

    # Center Gram matrices
    K_mean_row = K.mean(axis=0, keepdims=True)
    K_mean_col = K.mean(axis=1, keepdims=True)
    K_mean = K.mean()
    Kc = K - K_mean_row - K_mean_col + K_mean

    L_mean_row = L.mean(axis=0, keepdims=True)
    L_mean_col = L.mean(axis=1, keepdims=True)
    L_mean = L.mean()
    Lc = L - L_mean_row - L_mean_col + L_mean

    hsic = np.sum(Kc * Lc)
    var1 = np.sqrt(np.sum(Kc * Kc))
    var2 = np.sqrt(np.sum(Lc * Lc))
    denom = var1 * var2
    if (not np.isfinite(denom)) or denom == 0.0:
        return 0.0
    if not np.isfinite(hsic):
        return 0.0
    return float(hsic / denom)


def _center_gram(K: np.ndarray) -> np.ndarray:
    mean_row = K.mean(axis=0, keepdims=True)
    mean_col = K.mean(axis=1, keepdims=True)
    mean = K.mean()
    return K - mean_row - mean_col + mean


def _gram_rbf(X: np.ndarray, gamma=None, scale: float = 1.0) -> np.ndarray:
    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    if gamma is None:
        triu = sq_dists[np.triu_indices_from(sq_dists, k=1)]
        if triu.size == 0:
            gamma = 1.0
        else:
            median_sq = float(np.median(triu))
            if median_sq <= 0.0:
                gamma = 1.0
            else:
                gamma = scale / median_sq
    return np.exp(-gamma * sq_dists)


def cka_kernel(X: np.ndarray, Y: np.ndarray, gamma=None, scale: float = 1.0) -> float:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA requires the same number of rows, got {X.shape[0]} and {Y.shape[0]}."
        )
    K = _gram_rbf(X, gamma, scale)
    L = _gram_rbf(Y, gamma, scale)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    hsic = np.sum(Kc * Lc)
    var1 = np.sqrt(np.sum(Kc * Kc))
    var2 = np.sqrt(np.sum(Lc * Lc))
    denom = var1 * var2
    if (not np.isfinite(denom)) or denom == 0.0:
        return 0.0
    if not np.isfinite(hsic):
        return 0.0
    return float(hsic / denom)


def _r2_multioutput(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n_samples, n_features = Xc.shape
    XtX = Xc.T @ Xc
    XtX_reg = XtX + ridge * np.eye(n_features, dtype=Xc.dtype)
    Xty = Xc.T @ Yc
    try:
        beta = np.linalg.solve(XtX_reg, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX_reg) @ Xty
    Y_pred = Xc @ beta
    ss_res = np.sum((Yc - Y_pred) ** 2, axis=0)
    ss_tot = np.sum(Yc**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    r2 = np.clip(r2, 0.0, 1.0)
    return float(np.mean(r2))


def linear_regression_similarity(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> float:
    s_xy = _r2_multioutput(X, Y, ridge)
    s_yx = _r2_multioutput(Y, X, ridge)
    return 0.5 * (s_xy + s_yx)


def cca_similarity(X: np.ndarray, Y: np.ndarray, n_components=None) -> float:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CCA requires the same number of rows, got {X.shape[0]} and {Y.shape[0]}."
        )
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n_samples = Xc.shape[0]
    max_comp = min(n_samples - 1, Xc.shape[1], Yc.shape[1])
    if max_comp <= 0:
        return 0.0
    if n_components is None:
        n_components = min(10, max_comp)
    elif n_components > max_comp:
        n_components = max_comp
    cca = CCA(n_components=n_components, max_iter=500, scale=False)
    X_c, Y_c = cca.fit_transform(Xc, Yc)
    if X_c.ndim == 1:
        X_c = X_c[:, np.newaxis]
        Y_c = Y_c[:, np.newaxis]
    corrs = []
    for i in range(X_c.shape[1]):
        x = X_c[:, i]
        y = Y_c[:, i]
        num = float(np.dot(x, y))
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom == 0.0:
            corr = 0.0
        else:
            corr = num / denom
        corrs.append(corr * corr)
    if not corrs:
        return 0.0
    return float(np.mean(corrs))


def _svd_reduce(X: np.ndarray, var_threshold: float = 0.99) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.size == 0:
        return Xc
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if S.size == 0:
        return Xc
    var = S * S
    total = var.sum()
    if total <= 0.0:
        return Xc
    var_ratio = var / total
    cumulative = np.cumsum(var_ratio)
    k = int(np.searchsorted(cumulative, var_threshold) + 1)
    k = max(1, min(k, S.size))
    return U[:, :k] * S[:k]


def svcca_similarity(X: np.ndarray, Y: np.ndarray, var_threshold: float = 0.99) -> float:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"SVCCA requires the same number of rows, got {X.shape[0]} and {Y.shape[0]}."
        )
    Xr = _svd_reduce(X, var_threshold)
    Yr = _svd_reduce(Y, var_threshold)
    return cca_similarity(Xr, Yr)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare model fingerprints: compute similarities (CKA) "
            "between a base fingerprint and others in a directory."
        )
    )
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Path to the base fingerprint JSON file.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="output/comput_W",
        help=(
            "Directory containing fingerprint JSON files to compare (default: /output/comput_W)."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional labels for each fingerprint file (including the base). "
            "If omitted, file stems will be used."
        ),
    )
    parser.add_argument(
        "--cka",
        type=str,
        choices=["linear", "kernel"],
        default="linear",
        help="Type of CKA similarity to use when comparing fingerprints (default: linear).",
    )
    parser.add_argument(
        "--kernel_gamma_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for RBF kernel bandwidth (gamma = scale / median_sq; "
            "only used when --cka kernel)."
        ),
    )
    parser.add_argument(
        "--feature_threshold",
        type=float,
        default=None,
        help=(
            "If set, mask entries in fingerprint matrices below this value to 0 "
            "before computing CKA."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "If set, compare only a single layer (1-based index) of the fingerprints. "
            "If the requested layer exceeds some model's total L, the smallest L "
            "across models is used instead."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cka", "cca", "svcca", "linreg"],
        default="cka",
        help=(
            "Similarity measure for fingerprint comparison: 'cka' (default), "
            "'cca', 'svcca', or 'linreg'."
        ),
    )
    parser.add_argument(
        "--cca_per_sample",
        action="store_true",
        help=(
            "Only for per_sample_scores fingerprints with --mode cca: compute CCA per sample "
            "using each sample's [L,H] matrix (cropped to min L/H across the pair) and "
            "average over samples."
        ),
    )
    parser.add_argument(
        "--cca_single_sample",
        action="store_true",
        help=(
            "Only for per_sample_scores fingerprints with --mode cca: compute CCA once using a "
            "single sample's [L,H] matrix (cropped to min L/H across the pair)."
        ),
    )
    parser.add_argument(
        "--cca_transpose_mlh",
        action="store_true",
        help=(
            "Only for per_sample_scores fingerprints with --mode cca: reshape [M, L, H] into "
            "a matrix of shape [L*H, M] (rows=samples, cols=features) and compute CCA once."
        ),
    )
    parser.add_argument(
        "--cca_sample_index",
        type=int,
        default=1,
        help=("Which sample index to use (1-based) when --cca_single_sample is set (default: 1)."),
    )
    parser.add_argument(
        "--cca_components",
        type=int,
        default=None,
        help=(
            "Number of CCA components (n_components). If omitted, a default is chosen. "
            "Smaller values (e.g. 1-3) can reduce overfitting and avoid trivial ~1.0 scores."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cca_components is not None and args.cca_components <= 0:
        raise ValueError("--cca_components must be a positive integer.")

    if args.cca_sample_index <= 0:
        raise ValueError("--cca_sample_index must be a positive integer (1-based).")

    base_path = Path(args.base).resolve()
    dir_path = Path(args.dir)

    if not base_path.is_file():
        raise FileNotFoundError(f"Base fingerprint file not found: {base_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    # Collect all fingerprint JSON files in the directory
    all_files = sorted(dir_path.glob("fingerprint_*.json"))
    if not all_files:
        raise ValueError(f"No fingerprint_*.json files found in {dir_path}.")

    # Ensure base file is first; include it even if it is outside dir_path
    input_files = [base_path]
    for p in all_files:
        if p.resolve() != base_path:
            input_files.append(p)

    if args.labels is None:
        labels = [p.stem for p in input_files]
    else:
        if len(args.labels) != len(input_files):
            raise ValueError("Number of labels must match number of fingerprint files.")
        labels = args.labels

    print("Loading fingerprints ...")

    raw_arrays = []
    Ns = []
    repr_kinds = []
    Ls = []
    Hs = []

    for path, _label in zip(input_files, labels):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "L" not in data or "H" not in data:
            raise ValueError(f"File {path} must contain 'L' and 'H' fields.")

        L = int(data["L"])
        H = int(data["H"])

        if "per_sample_log_bucket" in data:
            # Log-distance bucket features: [M, L, H, 2K]
            bucket_arr = np.asarray(data["per_sample_log_bucket"], dtype=np.float64)
            if bucket_arr.ndim != 4:
                raise ValueError(
                    f"per_sample_log_bucket in file {path} must have shape [M, L, H, 2K], "
                    f"got {bucket_arr.shape}."
                )
            M, L_s, H_s, D = bucket_arr.shape
            if L_s != L or H_s != H:
                raise ValueError(
                    f"Shape mismatch in file {path}: per_sample_log_bucket has shape "
                    f"{bucket_arr.shape}, but L={L}, H={H}."
                )
            raw_arrays.append(bucket_arr)
            Ns.append(M)
            repr_kinds.append("per_sample_log_bucket")
        elif "per_sample_pool_lower" in data:
            # Pool lower triangular features: [M, L, H, D] - pooled lower tri per head
            pool_arr = np.asarray(data["per_sample_pool_lower"], dtype=np.float64)
            if pool_arr.ndim != 4:
                raise ValueError(
                    f"per_sample_pool_lower in file {path} must have shape [M, L, H, D], "
                    f"got {pool_arr.shape}."
                )
            M, L_s, H_s, D = pool_arr.shape
            if L_s != L or H_s != H:
                raise ValueError(
                    f"Shape mismatch in file {path}: per_sample_pool_lower has shape "
                    f"{pool_arr.shape}, but L={L}, H={H}."
                )
            raw_arrays.append(pool_arr)
            Ns.append(M)
            repr_kinds.append("per_sample_pool_lower")
        elif "per_sample_ars" in data:
            # ARS features: [M, L, H, D] - Attention Response Signature per head
            ars_arr = np.asarray(data["per_sample_ars"], dtype=np.float64)
            if ars_arr.ndim != 4:
                raise ValueError(
                    f"per_sample_ars in file {path} must have shape [M, L, H, D], "
                    f"got {ars_arr.shape}."
                )
            M, L_s, H_s, D = ars_arr.shape
            if L_s != L or H_s != H:
                raise ValueError(
                    f"Shape mismatch in file {path}: per_sample_ars has shape "
                    f"{ars_arr.shape}, but L={L}, H={H}."
                )
            raw_arrays.append(ars_arr)
            Ns.append(M)
            repr_kinds.append("per_sample_ars")
        elif "per_sample_svd" in data:
            # SVD features: [M, L, H, K] - top-k singular values per head
            svd_arr = np.asarray(data["per_sample_svd"], dtype=np.float64)
            if svd_arr.ndim != 4:
                raise ValueError(
                    f"per_sample_svd in file {path} must have shape [M, L, H, K], "
                    f"got {svd_arr.shape}."
                )
            M, L_s, H_s, K = svd_arr.shape
            if L_s != L or H_s != H:
                raise ValueError(
                    f"Shape mismatch in file {path}: per_sample_svd has shape "
                    f"{svd_arr.shape}, but L={L}, H={H}."
                )
            raw_arrays.append(svd_arr)
            Ns.append(M)
            repr_kinds.append("per_sample_svd")
        elif "per_sample_grids" in data:
            grids = np.asarray(data["per_sample_grids"], dtype=np.float64)
            if grids.ndim != 5:
                raise ValueError(
                    f"per_sample_grids in file {path} must have shape [M, L, H, S, S], "
                    f"got {grids.shape}."
                )
            M, L_s, H_s, S1, S2 = grids.shape
            if L_s != L or H_s != H or S1 != S2:
                raise ValueError(
                    f"Shape mismatch in file {path}: per_sample_grids has shape "
                    f"{grids.shape}, but L={L}, H={H}, and S must satisfy S1 == S2."
                )
            raw_arrays.append(grids)
            Ns.append(M)
            repr_kinds.append("per_sample_grid")
        elif "per_sample_scores" in data:
            per_sample = np.asarray(data["per_sample_scores"], dtype=np.float64)
            if per_sample.ndim != 3:
                raise ValueError(
                    f"per_sample_scores in file {path} must have shape [M, L, H], "
                    f"got {per_sample.shape}."
                )
            M, L_s, H_s = per_sample.shape
            if L_s != L or H_s != H:
                raise ValueError(
                    f"Shape mismatch in file {path}: per_sample_scores has shape "
                    f"{per_sample.shape}, but L={L}, H={H}."
                )
            raw_arrays.append(per_sample)
            Ns.append(M)
            repr_kinds.append("per_sample")
        else:
            if "fingerprint_vector" not in data:
                raise ValueError(f"File {path} does not contain 'fingerprint_vector'.")

            if "N_global" in data:
                N = int(data["N_global"])
            else:
                fp_len = int(
                    data.get(
                        "fingerprint_vector_length",
                        len(data["fingerprint_vector"]),
                    )
                )
                denom = L * H
                if fp_len % denom != 0:
                    raise ValueError(
                        "Cannot infer N for file "
                        f"{path}: fingerprint length {fp_len} is not divisible by "
                        f"L*H={denom}."
                    )
                N_sq = fp_len // denom
                N = int(round(np.sqrt(N_sq)))
                if N * N != N_sq:
                    raise ValueError(
                        "Cannot infer N for file "
                        f"{path}: L*H*N^2 does not match fingerprint length {fp_len}."
                    )

            vec = np.asarray(data["fingerprint_vector"], dtype=np.float64)
            raw_arrays.append(vec)
            Ns.append(N)
            repr_kinds.append("grid")

        Ls.append(L)
        Hs.append(H)

    repr_unique = sorted(set(repr_kinds))
    if len(repr_unique) != 1:
        raise ValueError(
            f"All fingerprints must use the same representation type, but got {repr_unique}."
        )

    if args.cca_per_sample and args.mode != "cca":
        print("[Warning] --cca_per_sample is only used when --mode cca; ignoring.")

    if args.cca_single_sample and args.mode != "cca":
        print("[Warning] --cca_single_sample is only used when --mode cca; ignoring.")

    if args.cca_transpose_mlh and args.mode != "cca":
        print("[Warning] --cca_transpose_mlh is only used when --mode cca; ignoring.")

    if args.cca_single_sample and args.cca_per_sample:
        print(
            "[Warning] Both --cca_single_sample and --cca_per_sample were set; using --cca_single_sample."
        )

    if args.cca_transpose_mlh and (args.cca_single_sample or args.cca_per_sample):
        print(
            "[Warning] --cca_transpose_mlh overrides --cca_single_sample/--cca_per_sample; using --cca_transpose_mlh."
        )

    N_unique = sorted(set(Ns))
    if len(N_unique) != 1:
        raise ValueError(
            "All fingerprints must have the same number of samples (rows) for CKA, but got "
            f"values {N_unique}."
        )

    # Determine layer index if requested (1-based in CLI, 0-based internally)
    layer_index = None
    if args.layer is not None:
        if args.layer < 1:
            raise ValueError("--layer must be a positive integer (1-based).")
        min_L = min(Ls)
        effective_layer = min(args.layer, min_L)
        if effective_layer != args.layer:
            print(
                f"Requested layer {args.layer} exceeds at least one model's total L; "
                f"using smallest L across models = {min_L}, so layer {effective_layer} "
                "will be used instead."
            )
        layer_index = effective_layer - 1
        print(f"Restricting comparison to layer {effective_layer} (1-based).")

    # Build matrices according to representation and (optional) layer selection
    matrices = []
    for arr, kind, L, H, num_rows, label in zip(raw_arrays, repr_kinds, Ls, Hs, Ns, labels):
        if kind == "per_sample_log_bucket":
            bucket_arr = arr
            D = bucket_arr.shape[3]
            if layer_index is None:
                mat = bucket_arr.reshape(num_rows, L * H * D)
                print(
                    f"Model {label}: L={L}, H={H}, D={D}, M={num_rows}, "
                    f"per-sample Log-Bucket shape = {bucket_arr.shape}, "
                    f"matrix shape = {mat.shape}"
                )
            else:
                mat = bucket_arr[:, layer_index, :, :].reshape(num_rows, H * D)
                print(
                    f"Model {label}: L={L}, H={H}, D={D}, M={num_rows}, "
                    f"per-sample Log-Bucket shape = {bucket_arr.shape}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )
        elif kind == "per_sample_pool_lower":
            # Pool lower features: [M, L, H, D] -> flatten to [M, L*H*D]
            pool_arr = arr
            D = pool_arr.shape[3]
            if layer_index is None:
                mat = pool_arr.reshape(num_rows, L * H * D)
                print(
                    f"Model {label}: L={L}, H={H}, D={D}, M={num_rows}, "
                    f"per-sample Pool Lower shape = {pool_arr.shape}, "
                    f"matrix shape = {mat.shape}"
                )
            else:
                mat = pool_arr[:, layer_index, :, :].reshape(num_rows, H * D)
                print(
                    f"Model {label}: L={L}, H={H}, D={D}, M={num_rows}, "
                    f"per-sample Pool Lower shape = {pool_arr.shape}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )
        elif kind == "per_sample_ars":
            # ARS features: [M, L, H, D] -> flatten to [M, L*H*D]
            ars_arr = arr
            D = ars_arr.shape[3]
            if layer_index is None:
                mat = ars_arr.reshape(num_rows, L * H * D)
                print(
                    f"Model {label}: L={L}, H={H}, D={D}, M={num_rows}, "
                    f"per-sample ARS shape = {ars_arr.shape}, "
                    f"matrix shape = {mat.shape}"
                )
            else:
                mat = ars_arr[:, layer_index, :, :].reshape(num_rows, H * D)
                print(
                    f"Model {label}: L={L}, H={H}, D={D}, M={num_rows}, "
                    f"per-sample ARS shape = {ars_arr.shape}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )
        elif kind == "per_sample_svd":
            # SVD features: [M, L, H, K] -> flatten to [M, L*H*K]
            svd_arr = arr
            K = svd_arr.shape[3]
            if layer_index is None:
                mat = svd_arr.reshape(num_rows, L * H * K)
                print(
                    f"Model {label}: L={L}, H={H}, K={K}, M={num_rows}, "
                    f"per-sample SVD shape = {svd_arr.shape}, "
                    f"matrix shape = {mat.shape}"
                )
            else:
                mat = svd_arr[:, layer_index, :, :].reshape(num_rows, H * K)
                print(
                    f"Model {label}: L={L}, H={H}, K={K}, M={num_rows}, "
                    f"per-sample SVD shape = {svd_arr.shape}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )
        elif kind == "per_sample":
            per_sample = arr
            if layer_index is None:
                mat = per_sample.reshape(num_rows, L * H)
                print(
                    f"Model {label}: L={L}, H={H}, M={num_rows}, "
                    f"per-sample scores shape = {per_sample.shape}, "
                    f"matrix shape = {mat.shape}"
                )
            else:
                mat = per_sample[:, layer_index, :].reshape(num_rows, H)
                print(
                    f"Model {label}: L={L}, H={H}, M={num_rows}, "
                    f"per-sample scores shape = {per_sample.shape}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )
        elif kind == "per_sample_grid":
            grids = arr
            if grids.ndim != 5:
                raise ValueError(
                    f"per_sample_grids for model {label} must have shape [M, L, H, S, S], "
                    f"got {grids.shape}."
                )
            _, L_g, H_g, S1, S2 = grids.shape
            if L_g != L or H_g != H or S1 != S2:
                raise ValueError(
                    f"Shape mismatch for model {label}: per_sample_grids has shape "
                    f"{grids.shape}, but L={L}, H={H}, and S must satisfy S1 == S2."
                )
            S = S1
            if layer_index is None:
                mat = grids.reshape(num_rows, L * H * S * S)
                print(
                    f"Model {label}: L={L}, H={H}, S={S}, M={num_rows}, "
                    f"per-sample grids shape = {grids.shape}, "
                    f"matrix shape = {mat.shape}"
                )
            else:
                if not (0 <= layer_index < L_g):
                    raise ValueError(
                        f"layer_index {layer_index} out of range for model {label} with L={L_g}."
                    )
                layer_grids = grids[:, layer_index, :, :, :]  # [M, H, S, S]
                mat = layer_grids.reshape(num_rows, H * S * S)
                print(
                    f"Model {label}: L={L}, H={H}, S={S}, M={num_rows}, "
                    f"per-sample grids shape = {grids.shape}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )
        else:
            vec = arr
            if layer_index is None:
                mat = fingerprint_to_matrix(vec, L, H, num_rows)
                print(
                    f"Model {label}: L={L}, H={H}, N={num_rows}, "
                    f"vector length = {vec.shape[0]}, matrix shape = {mat.shape}"
                )
            else:
                mat = fingerprint_layer_to_matrix(vec, L, H, num_rows, layer_index)
                print(
                    f"Model {label}: L={L}, H={H}, N={num_rows}, "
                    f"vector length = {vec.shape[0]}, "
                    f"using layer {layer_index + 1}, matrix shape = {mat.shape}"
                )

        matrices.append(mat)

    num_models = len(matrices)
    print(
        f"Each fingerprint represented as matrix of shape: {matrices[0].shape} (samples x features)"
    )

    # Optionally mask small features in the fingerprint matrices
    if args.feature_threshold is not None:
        thr = float(args.feature_threshold)
        print(
            f"Applying feature threshold: values < {thr} in fingerprint matrices "
            "will be set to 0 before CKA."
        )
        for i in range(num_models):
            mat = matrices[i]
            mask = mat < thr
            if np.any(mask):
                mat = mat.copy()
                mat[mask] = 0.0
                matrices[i] = mat

    for i in range(num_models):
        matrices[i] = _sanitize_matrix(matrices[i], labels[i])

    # Compute pairwise CKA similarities
    if args.mode == "cka":
        print(f"\nPairwise similarities ({args.cka} CKA):")
    else:
        print(f"\nPairwise similarities (mode={args.mode}):")
    base_matrix = matrices[0]
    base_label = labels[0]
    for j in range(1, num_models):
        Xj = matrices[j]
        if args.mode == "cka":
            if args.cka == "linear":
                sim_val = cka_linear(base_matrix, Xj)
            elif args.cka == "kernel":
                sim_val = cka_kernel(base_matrix, Xj, scale=args.kernel_gamma_scale)
            else:
                raise ValueError(f"Unknown CKA mode: {args.cka}")
            metric_name = "CKA"
        elif args.mode == "linreg":
            sim_val = linear_regression_similarity(base_matrix, Xj)
            metric_name = "LINREG"
        elif args.mode == "cca":
            if (
                args.cca_transpose_mlh
                and repr_kinds[0] == "per_sample"
                and repr_kinds[j] == "per_sample"
            ):
                if layer_index is not None:
                    sim_val = cca_similarity(base_matrix, Xj, n_components=args.cca_components)
                    metric_name = "CCA"
                else:
                    base_arr = np.asarray(raw_arrays[0], dtype=np.float64)
                    other_arr = np.asarray(raw_arrays[j], dtype=np.float64)
                    if base_arr.ndim != 3 or other_arr.ndim != 3:
                        raise ValueError(
                            "--cca_transpose_mlh requires per_sample_scores arrays with shape [M, L, H]."
                        )
                    M0, L0, H0 = base_arr.shape
                    M1, L1, H1 = other_arr.shape
                    if M0 != M1:
                        raise ValueError(
                            f"--cca_transpose_mlh requires same M across models, got {M0} and {M1}."
                        )
                    Lm = min(L0, L1)
                    Hm = min(H0, H1)
                    Xmat = base_arr[:, :Lm, :Hm].transpose(1, 2, 0).reshape(Lm * Hm, M0)
                    Ymat = other_arr[:, :Lm, :Hm].transpose(1, 2, 0).reshape(Lm * Hm, M0)
                    Xmat = np.where(np.isfinite(Xmat), Xmat, 0.0)
                    Ymat = np.where(np.isfinite(Ymat), Ymat, 0.0)
                    sim_val = cca_similarity(Xmat, Ymat, n_components=args.cca_components)
                    metric_name = "CCA_LH_by_M"
            elif (
                args.cca_single_sample
                and repr_kinds[0] == "per_sample"
                and repr_kinds[j] == "per_sample"
            ):
                if layer_index is not None:
                    sim_val = cca_similarity(base_matrix, Xj, n_components=args.cca_components)
                    metric_name = "CCA"
                else:
                    base_arr = np.asarray(raw_arrays[0], dtype=np.float64)
                    other_arr = np.asarray(raw_arrays[j], dtype=np.float64)
                    if base_arr.ndim != 3 or other_arr.ndim != 3:
                        raise ValueError(
                            "--cca_single_sample requires per_sample_scores arrays with shape [M, L, H]."
                        )
                    M0, L0, H0 = base_arr.shape
                    M1, L1, H1 = other_arr.shape
                    if M0 != M1:
                        raise ValueError(
                            f"--cca_single_sample requires same M across models, got {M0} and {M1}."
                        )
                    m = args.cca_sample_index - 1
                    if not (0 <= m < M0):
                        raise ValueError(
                            f"--cca_sample_index {args.cca_sample_index} out of range for M={M0}."
                        )
                    Lm = min(L0, L1)
                    Hm = min(H0, H1)
                    Xm = base_arr[m, :Lm, :Hm]
                    Ym = other_arr[m, :Lm, :Hm]
                    Xm = np.where(np.isfinite(Xm), Xm, 0.0)
                    Ym = np.where(np.isfinite(Ym), Ym, 0.0)
                    sim_val = cca_similarity(Xm, Ym, n_components=args.cca_components)
                    metric_name = "CCA_SINGLE_SAMPLE"
            elif (
                args.cca_per_sample
                and repr_kinds[0] == "per_sample"
                and repr_kinds[j] == "per_sample"
            ):
                if layer_index is not None:
                    sim_val = cca_similarity(base_matrix, Xj, n_components=args.cca_components)
                    metric_name = "CCA"
                else:
                    base_arr = np.asarray(raw_arrays[0], dtype=np.float64)
                    other_arr = np.asarray(raw_arrays[j], dtype=np.float64)
                    if base_arr.ndim != 3 or other_arr.ndim != 3:
                        raise ValueError(
                            "--cca_per_sample requires per_sample_scores arrays with shape [M, L, H]."
                        )
                    M0, L0, H0 = base_arr.shape
                    M1, L1, H1 = other_arr.shape
                    if M0 != M1:
                        raise ValueError(
                            f"--cca_per_sample requires same M across models, got {M0} and {M1}."
                        )
                    Lm = min(L0, L1)
                    Hm = min(H0, H1)
                    vals = []
                    for m in range(M0):
                        Xm = base_arr[m, :Lm, :Hm]
                        Ym = other_arr[m, :Lm, :Hm]
                        Xm = np.where(np.isfinite(Xm), Xm, 0.0)
                        Ym = np.where(np.isfinite(Ym), Ym, 0.0)
                        vals.append(cca_similarity(Xm, Ym, n_components=args.cca_components))
                    sim_val = float(np.mean(vals)) if vals else 0.0
                    metric_name = "CCA_PER_SAMPLE"
            else:
                sim_val = cca_similarity(base_matrix, Xj, n_components=args.cca_components)
                metric_name = "CCA"
        elif args.mode == "svcca":
            sim_val = svcca_similarity(base_matrix, Xj)
            metric_name = "SVCCA"
        else:
            raise ValueError(f"Unknown similarity mode: {args.mode}")

        if args.mode == "cka":
            print(f"{base_label} vs {labels[j]}: CKA = {sim_val:.4f}")
        else:
            print(f"{base_label} vs {labels[j]}: {metric_name} = {sim_val:.4f}")


if __name__ == "__main__":
    main()
