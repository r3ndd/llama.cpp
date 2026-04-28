from __future__ import annotations

import numpy as np

from .types import MatrixMetrics


SPECTRAL_ENERGY_RANK_FRACTIONS: tuple[float, ...] = tuple(x / 100.0 for x in range(5, 100, 5))


def rank_from_fraction(m: int, n: int, rank_frac: float) -> int:
    return max(1, round(rank_frac * min(m, n)))


def analyze_matrix(matrix: np.ndarray, rank_frac: float) -> MatrixMetrics:
    warnings: list[str] = []

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")

    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values")

    m, n = matrix.shape
    rank_used = rank_from_fraction(m, n, rank_frac)

    s = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)

    s2 = np.square(s, dtype=np.float64)
    denom = float(np.sum(np.square(s2, dtype=np.float64), dtype=np.float64))
    if denom == 0.0:
        participation_ratio = 0.0
        warnings.append("zero_singular_values_denominator")
    else:
        numer = float(np.square(np.sum(s2, dtype=np.float64), dtype=np.float64))
        participation_ratio = numer / denom

    total_spectral_energy = float(np.sum(s2, dtype=np.float64))
    if total_spectral_energy == 0.0:
        explained_spectral_energy_rank_fractions = [0.0 for _ in SPECTRAL_ENERGY_RANK_FRACTIONS]
        warnings.append("zero_total_spectral_energy")
    else:
        cumulative_s2 = np.cumsum(s2, dtype=np.float64)
        explained_spectral_energy_rank_fractions = []
        for frac in SPECTRAL_ENERGY_RANK_FRACTIONS:
            frac_rank = rank_from_fraction(m, n, frac)
            retained = float(cumulative_s2[frac_rank - 1])
            explained_spectral_energy_rank_fractions.append(retained / total_spectral_energy)

    fro_norm = float(np.sqrt(total_spectral_energy))

    if not np.isfinite(participation_ratio):
        raise ValueError("Computed non-finite participation ratio")
    if not np.isfinite(fro_norm):
        raise ValueError("Computed non-finite Frobenius norm")
    if not all(np.isfinite(x) for x in explained_spectral_energy_rank_fractions):
        raise ValueError("Computed non-finite explained spectral energy fractions")

    return MatrixMetrics(
        m=m,
        n=n,
        rank_used=rank_used,
        singular_value_count=int(s.shape[0]),
        participation_ratio=participation_ratio,
        explained_spectral_energy_rank_fractions=explained_spectral_energy_rank_fractions,
        fro_norm=fro_norm,
        analysis_warnings=warnings,
    )
