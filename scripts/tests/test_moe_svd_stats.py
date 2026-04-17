from __future__ import annotations

from collections import Counter

import pytest

from moe_svd.stats import compute_summary
from moe_svd.types import FailedMatrix, PerMatrixRecord


def _record(pr: float, cos: float) -> PerMatrixRecord:
    return PerMatrixRecord(
        tensor="t",
        layer=0,
        expert=0,
        role="w1",
        tensor_type="Q4_K",
        shape=(4, 4),
        rank_used=1,
        singular_value_count=4,
        participation_ratio=pr,
        cosine_similarity_lowrank=cos,
        fro_norm=1.0,
        elapsed_seconds=0.1,
        warnings=[],
    )


def test_compute_summary_counts_and_percentiles() -> None:
    per_matrix = [_record(1.0, 0.9), _record(2.0, 0.8), _record(3.0, 0.7)]
    failed = [
        FailedMatrix(tensor="x", layer=1, expert=2, role=None, reason="MemoryError"),
        FailedMatrix(tensor="y", layer=1, expert=3, role=None, reason="MemoryError"),
    ]

    summary = compute_summary(
        per_matrix,
        total_tensors=10,
        candidates=5,
        skipped_reasons=Counter({"non_2d_tensor": 2}),
        failed=failed,
    )

    assert summary.participation_ratio.count == 3
    assert summary.participation_ratio.mean == 2.0
    assert summary.cosine_similarity.mean == pytest.approx(0.8)
    assert summary.counts["total_tensors"] == 10
    assert summary.counts["candidates"] == 5
    assert summary.counts["analyzed"] == 3
    assert summary.counts["failed_by_reason"]["MemoryError"] == 2


def test_compute_summary_empty() -> None:
    summary = compute_summary(
        [],
        total_tensors=0,
        candidates=0,
        skipped_reasons=Counter(),
        failed=[],
    )

    assert summary.participation_ratio.count == 0
    assert summary.cosine_similarity.count == 0
    assert summary.counts["analyzed"] == 0
