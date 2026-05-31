"""Tests for src/boundary_tracker.py — four unit tests as specified."""
import numpy as np
import pytest
from src.boundary_tracker import find_roots_in_interval


def test_synthetic_sin_roots():
    """F(x) = sin(2πx) on [0,1]: tracker must find roots near x=0, 0.5, 1."""
    F = lambda x: np.sin(2 * np.pi * np.asarray(x, dtype=float))
    result = find_roots_in_interval(F, 0.0, 1.0, n_plot=2000)
    assert result['n_roots'] == 3, f"Expected 3 roots, got {result['n_roots']}"
    assert abs(result['all_roots'][0] - 0.0) < 0.01
    assert abs(result['all_roots'][1] - 0.5) < 0.01
    assert abs(result['all_roots'][2] - 1.0) < 0.01


def test_no_root():
    """F(x) = 1 + sin(x) on [0,1]: always positive, tracker returns 0 roots and NaN s."""
    F = lambda x: 1.0 + np.sin(np.asarray(x, dtype=float))
    result = find_roots_in_interval(F, 0.0, 1.0, n_plot=1000)
    assert result['n_roots'] == 0
    assert np.isnan(result['s'])


def test_transversality():
    """F(x) = x - 0.5: root at 0.5 with κ = |∂_x F| = 1.0."""
    F = lambda x: np.asarray(x, dtype=float) - 0.5
    result = find_roots_in_interval(F, 0.0, 1.0, n_plot=1000)
    assert result['n_roots'] == 1
    assert abs(result['s'] - 0.5) < 1e-8
    assert abs(result['kappa'] - 1.0) < 1e-5


def test_multiple_roots_warning(capsys):
    """F with 3 sign changes: n_roots=3 recorded and a WARNING is printed."""
    # (x-0.25)(x-0.5)(x-0.75) has exactly 3 simple roots in (0,1)
    F = lambda x: (
        (np.asarray(x, dtype=float) - 0.25)
        * (np.asarray(x, dtype=float) - 0.50)
        * (np.asarray(x, dtype=float) - 0.75)
    )
    result = find_roots_in_interval(F, 0.0, 1.0, n_plot=2000)
    assert result['n_roots'] == 3, f"Expected 3 roots, got {result['n_roots']}"
    assert abs(result['all_roots'][0] - 0.25) < 1e-5
    assert abs(result['all_roots'][1] - 0.50) < 1e-5
    assert abs(result['all_roots'][2] - 0.75) < 1e-5
    # The WARNING message must be printed (not silently dropped to 1 root)
    captured = capsys.readouterr()
    assert 'WARNING' in captured.out
