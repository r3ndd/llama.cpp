from __future__ import annotations

import numpy as np

from moe_svd.svd_metrics import SPECTRAL_ENERGY_RANK_FRACTIONS, analyze_matrix, rank_from_fraction


def test_rank_from_fraction_rounding_and_minimum() -> None:
    assert rank_from_fraction(10, 20, 0.05) == 1
    assert rank_from_fraction(100, 100, 0.05) == 5
    assert rank_from_fraction(3, 7, 0.05) == 1


def test_participation_ratio_on_known_diagonal_spectrum() -> None:
    s = np.array([4.0, 3.0, 0.0, 0.0], dtype=np.float64)
    w = np.diag(s)

    metrics = analyze_matrix(w, rank_frac=0.5)
    expected_pr = ((4.0**2 + 3.0**2) ** 2) / ((4.0**4 + 3.0**4))

    assert np.isclose(metrics.participation_ratio, expected_pr)


def test_zero_matrix_guards() -> None:
    w = np.zeros((8, 8), dtype=np.float32)
    metrics = analyze_matrix(w, rank_frac=0.25)

    assert metrics.participation_ratio == 0.0
    assert len(metrics.explained_spectral_energy_rank_fractions) == len(SPECTRAL_ENERGY_RANK_FRACTIONS)
    assert all(x == 0.0 for x in metrics.explained_spectral_energy_rank_fractions)
    assert "zero_singular_values_denominator" in metrics.analysis_warnings
    assert "zero_total_spectral_energy" in metrics.analysis_warnings


def test_explained_spectral_energy_uses_singular_values() -> None:
    w = np.diag(np.array([5.0, 4.0, 3.0, 0.0], dtype=np.float64))
    metrics = analyze_matrix(w, rank_frac=0.5)

    idx_50 = list(SPECTRAL_ENERGY_RANK_FRACTIONS).index(0.50)
    expected_50 = (5.0**2 + 4.0**2) / (5.0**2 + 4.0**2 + 3.0**2)
    assert np.isclose(metrics.explained_spectral_energy_rank_fractions[idx_50], expected_50)

    idx_05 = list(SPECTRAL_ENERGY_RANK_FRACTIONS).index(0.05)
    expected_05 = (5.0**2) / (5.0**2 + 4.0**2 + 3.0**2)
    assert np.isclose(metrics.explained_spectral_energy_rank_fractions[idx_05], expected_05)
