"""Tests for similarity computation."""

import numpy as np
import pytest

from attndiff.core.similarity import fingerprint_to_matrix, linear_cka


def test_linear_cka_identical():
    X = np.random.rand(10, 20)
    similarity = linear_cka(X, X)
    assert abs(similarity - 1.0) < 1e-6


def test_linear_cka_orthogonal():
    X = np.array([[1.0, 0.0], [0.0, 0.0]])
    Y = np.array([[0.0, 1.0], [0.0, 0.0]])
    similarity = linear_cka(X, Y)
    assert abs(similarity) < 1e-6


def test_fingerprint_to_matrix():
    L, H, N = 2, 2, 3
    vec = np.arange(L * H * N * N, dtype=np.float64)

    mat = fingerprint_to_matrix(vec, L, H, N)

    assert mat.shape == (N, L * H * N)


def test_fingerprint_to_matrix_mismatch():
    vec = np.arange(10, dtype=np.float64)

    with pytest.raises(ValueError, match="Fingerprint length mismatch"):
        fingerprint_to_matrix(vec, L=2, H=2, N=3)
