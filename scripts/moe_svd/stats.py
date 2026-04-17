from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .types import FailedMatrix, PerMatrixRecord, SummaryDistribution, SummaryStats


def _distribution(values: list[float]) -> SummaryDistribution:
    if not values:
        return SummaryDistribution(
            count=0,
            mean=0.0,
            median=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            p10=0.0,
            p25=0.0,
            p75=0.0,
            p90=0.0,
        )

    arr = np.asarray(values, dtype=np.float64)
    return SummaryDistribution(
        count=int(arr.shape[0]),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
    )


def compute_summary(
    per_matrix: list[PerMatrixRecord],
    *,
    total_tensors: int,
    candidates: int,
    skipped_reasons: Counter[str],
    failed: list[FailedMatrix],
) -> SummaryStats:
    pr = [r.participation_ratio for r in per_matrix]
    cs = [r.cosine_similarity_lowrank for r in per_matrix]

    failed_reasons: Counter[str] = Counter(r.reason for r in failed)

    counts: dict[str, Any] = {
        "total_tensors": total_tensors,
        "candidates": candidates,
        "analyzed": len(per_matrix),
        "skipped_by_reason": dict(sorted(skipped_reasons.items())),
        "failed_by_reason": dict(sorted(failed_reasons.items())),
    }

    return SummaryStats(
        participation_ratio=_distribution(pr),
        cosine_similarity=_distribution(cs),
        counts=counts,
    )
