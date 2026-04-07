"""Tests for fingerprint computation."""

import numpy as np
import pytest

from attndiff.core.fingerprint import compute_fingerprint


def test_compute_fingerprint_basic():
    L, H, N, M = 2, 2, 4, 3
    
    original_data = []
    corrupted_data = []
    
    for _ in range(M):
        sample_orig = []
        sample_corr = []
        for _ in range(L):
            layer_orig = []
            layer_corr = []
            for _ in range(H):
                layer_orig.append(np.random.rand(N, N).tolist())
                layer_corr.append(np.random.rand(N, N).tolist())
            sample_orig.append(layer_orig)
            sample_corr.append(layer_corr)
        original_data.append(sample_orig)
        corrupted_data.append(sample_corr)
    
    result = compute_fingerprint(original_data, corrupted_data, mode="diff")
    
    assert "fingerprint_vector" in result
    assert result["L"] == L
    assert result["H"] == H
    assert result["N"] == N
    assert result["M"] == M
    assert len(result["fingerprint_vector"]) == L * H * N * N


def test_compute_fingerprint_mode_orig():
    L, H, N, M = 1, 1, 2, 1
    
    original_data = [[[[1.0, 2.0], [3.0, 4.0]]]]
    corrupted_data = [[[[0.0, 0.0], [0.0, 0.0]]]]
    
    result = compute_fingerprint(original_data, corrupted_data, mode="orig")
    
    assert result["fingerprint_vector"] == [1.0, 2.0, 3.0, 4.0]


def test_compute_fingerprint_mismatch():
    original_data = [[[[1.0]]]]
    corrupted_data = [[[[1.0]]], [[[2.0]]]]
    
    with pytest.raises(ValueError, match="Mismatch in sample count"):
        compute_fingerprint(original_data, corrupted_data)
