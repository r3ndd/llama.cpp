from __future__ import annotations

import json
from pathlib import Path

from .types import Report, SummaryDistribution


class ReportingError(RuntimeError):
    """Raised when report serialization/output fails."""


def _fmt_dist(name: str, dist: SummaryDistribution) -> str:
    return (
        f"{name}: count={dist.count} mean={dist.mean:.6f} median={dist.median:.6f} "
        f"std={dist.std:.6f} min={dist.min:.6f} max={dist.max:.6f} "
        f"p10={dist.p10:.6f} p25={dist.p25:.6f} p75={dist.p75:.6f} p90={dist.p90:.6f}"
    )


def print_cli_summary(report: Report, quiet: bool) -> None:
    if quiet:
        return

    run = report.run
    counts = report.summary.counts

    print("=== MoE Expert SVD Compressibility Summary ===")
    print(f"Model: {run['model_spec']}")
    print(f"Resolved file: {run['resolved_path']}")
    print(f"dtype={run['dtype']} rank_frac={run['rank_fraction']:.6f} mode={run['fidelity_mode']}")
    print(
        "Discovery counts: "
        f"total_tensors={counts['total_tensors']} candidates={counts['candidates']} "
        f"analyzed={counts['analyzed']} failed={len(report.failed_matrices)}",
    )
    print(_fmt_dist("Participation ratio", report.summary.participation_ratio))
    print(_fmt_dist("Cosine similarity", report.summary.cosine_similarity))

    if counts["skipped_by_reason"]:
        print("Skipped by reason:")
        for reason, n in counts["skipped_by_reason"].items():
            print(f"  - {reason}: {n}")

    if counts["failed_by_reason"]:
        print("Failed by reason:")
        for reason, n in counts["failed_by_reason"].items():
            print(f"  - {reason}: {n}")


def write_json_report(report: Report, out_path: str) -> None:
    path = Path(out_path).expanduser().resolve()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise ReportingError(f"Failed to write JSON report at '{path}': {exc}") from exc
