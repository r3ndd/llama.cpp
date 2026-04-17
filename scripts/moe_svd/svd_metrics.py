from __future__ import annotations

import math

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

    wr = (u[:, :rank_used] * s[:rank_used]) @ vt[:rank_used, :]

    flat_w = matrix.ravel()
    flat_wr = wr.ravel()

    norm_w = float(np.linalg.norm(flat_w))
    norm_wr = float(np.linalg.norm(flat_wr))
    if norm_w == 0.0 or norm_wr == 0.0:
        cosine_similarity = 0.0
        warnings.append("zero_norm_for_cosine")
    else:
        cosine_similarity = float(np.dot(flat_w, flat_wr) / (norm_w * norm_wr))
        if not math.isfinite(cosine_similarity):
            cosine_similarity = 0.0
            warnings.append("non_finite_cosine")
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    fro_norm = float(np.linalg.norm(matrix, ord="fro"))

    if not math.isfinite(participation_ratio):
        raise ValueError("Computed non-finite participation ratio")

    return MatrixMetrics(
        m=m,
        n=n,
        rank_used=rank_used,
        singular_value_count=int(s.shape[0]),
        participation_ratio=participation_ratio,
        cosine_similarity_lowrank=cosine_similarity,
        fro_norm=fro_norm,
        analysis_warnings=warnings,
    )
