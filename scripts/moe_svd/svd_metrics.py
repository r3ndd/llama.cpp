from __future__ import annotations

import numpy as np

from .types import MatrixMetrics


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

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)

    s2 = np.square(s, dtype=np.float64)
    denom = float(np.sum(np.square(s2, dtype=np.float64), dtype=np.float64))
    if denom == 0.0:
        participation_ratio = 0.0
        warnings.append("zero_singular_values_denominator")
    else:
        numer = float(np.square(np.sum(s2, dtype=np.float64), dtype=np.float64))
        participation_ratio = numer / denom

    total_spectral_energy = float(np.sum(s2, dtype=np.float64))
    retained_spectral_energy = float(np.sum(s2[:rank_used], dtype=np.float64))
    if total_spectral_energy == 0.0:
        explained_spectral_energy_rank_r = 0.0
        warnings.append("zero_total_spectral_energy")
    else:
        explained_spectral_energy_rank_r = retained_spectral_energy / total_spectral_energy

    fro_norm = float(np.linalg.norm(matrix, ord="fro"))

    if not np.isfinite(participation_ratio):
        raise ValueError("Computed non-finite participation ratio")
    if not np.isfinite(explained_spectral_energy_rank_r):
        raise ValueError("Computed non-finite explained spectral energy")

    return MatrixMetrics(
        m=m,
        n=n,
        rank_used=rank_used,
        singular_value_count=int(s.shape[0]),
        participation_ratio=participation_ratio,
        explained_spectral_energy_rank_r=explained_spectral_energy_rank_r,
        fro_norm=fro_norm,
        analysis_warnings=warnings,
    )
