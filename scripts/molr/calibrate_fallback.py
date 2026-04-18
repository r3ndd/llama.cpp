#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.types import MOLR_EXPERT_VALIDATION_SCHEMA_VERSION, MOLR_THRESHOLDS_SCHEMA_VERSION


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrCalibrationError(RuntimeError):
    """Raised when fallback calibration cannot proceed."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrCalibrationError(f"Failed reading JSON '{path}': {exc}") from exc


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrCalibrationError(f"Failed writing JSON '{path}': {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-3 fallback calibration over per-expert validation summaries. "
            "Builds threshold lookup table and cache-candidate recommendations."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Path to Phase-2 checkpoints directory (recorded for traceability).",
    )
    parser.add_argument(
        "--validation-dir",
        default="",
        help=(
            "Directory with per-expert validation JSON files (molr_validation_<layer>_<expert>.json). "
            "If omitted, defaults to sibling '<checkpoints-parent>/validation'."
        ),
    )
    parser.add_argument(
        "--validation-glob",
        default="molr_validation_*.json",
        help="Validation filename glob inside --validation-dir. Default: molr_validation_*.json",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model-spec guard; if provided, validation rows must match.",
    )
    parser.add_argument(
        "--quality-profiles",
        default="balanced:0.90,quality:0.95,strict:0.98",
        help=(
            "Comma-separated profile_name:quality_proxy_min targets. "
            "Example: balanced:0.90,quality:0.95"
        ),
    )
    parser.add_argument(
        "--top-cache-candidates",
        type=int,
        default=32,
        help="Maximum recommended experts for full-weight cache list. Default: 32.",
    )
    parser.add_argument("--out-json", required=True, help="Output path for molr_thresholds.json.")

    args = parser.parse_args(argv)
    if args.top_cache_candidates < 0:
        parser.error("--top-cache-candidates must be >= 0")
    return args


def _parse_quality_profiles(raw: str) -> dict[str, float]:
    profiles: dict[str, float] = {}
    if not raw.strip():
        raise MolrCalibrationError("--quality-profiles cannot be empty")
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise MolrCalibrationError(
                f"Invalid --quality-profiles token '{token}', expected name:value format.",
            )
        name_raw, value_raw = token.split(":", 1)
        name = name_raw.strip()
        if not name:
            raise MolrCalibrationError(f"Invalid profile name in token '{token}'")
        try:
            value = float(value_raw)
        except Exception as exc:
            raise MolrCalibrationError(f"Invalid profile value in token '{token}'") from exc
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise MolrCalibrationError(
                f"Profile '{name}' quality target must be in (0,1], got {value}",
            )
        profiles[name] = value
    if not profiles:
        raise MolrCalibrationError("--quality-profiles did not yield any profiles")
    return profiles


def _validation_files(args: argparse.Namespace, checkpoints_dir: Path) -> tuple[Path, list[Path]]:
    if args.validation_dir:
        validation_dir = Path(args.validation_dir).expanduser().resolve()
    else:
        validation_dir = checkpoints_dir.parent / "validation"
    if not validation_dir.is_dir():
        raise MolrCalibrationError(f"Validation directory does not exist: '{validation_dir}'")
    files = sorted(validation_dir.glob(args.validation_glob))
    if not files:
        raise MolrCalibrationError(
            f"No validation JSON files matched glob '{args.validation_glob}' in '{validation_dir}'",
        )
    return validation_dir, files


def _load_rows(
    files: list[Path],
    *,
    expected_model: str,
) -> tuple[str, list[dict[str, Any]]]:
    model_seen: str | None = None
    rows: list[dict[str, Any]] = []

    for path in files:
        payload = _load_json(path)
        schema = str(payload.get("schema_version") or "")
        if schema != MOLR_EXPERT_VALIDATION_SCHEMA_VERSION:
            raise MolrCalibrationError(
                f"Validation schema mismatch in '{path}': got '{schema}', "
                f"expected '{MOLR_EXPERT_VALIDATION_SCHEMA_VERSION}'",
            )

        model_spec = str(payload.get("model_spec") or "")
        if not model_spec:
            raise MolrCalibrationError(f"Validation file missing model_spec: '{path}'")
        if expected_model and model_spec != expected_model:
            raise MolrCalibrationError(
                f"Model mismatch in '{path}': got '{model_spec}', expected '{expected_model}'",
            )
        if model_seen is None:
            model_seen = model_spec
        elif model_seen != model_spec:
            raise MolrCalibrationError(
                f"Validation files mix model specs: '{model_seen}' and '{model_spec}'",
            )

        metrics = payload.get("validation_metrics", {})
        try:
            pred_error_mean = float(metrics.get("pred_error_mean", 0.0))
            true_error_mean = float(metrics.get("true_error_mean", 0.0))
        except Exception as exc:
            raise MolrCalibrationError(
                f"Invalid validation metric type in '{path}'",
            ) from exc
        if not math.isfinite(pred_error_mean) or pred_error_mean < 0.0:
            raise MolrCalibrationError(f"Invalid pred_error_mean in '{path}'")
        if not math.isfinite(true_error_mean) or true_error_mean < 0.0:
            raise MolrCalibrationError(f"Invalid true_error_mean in '{path}'")

        try:
            layer = int(payload.get("layer"))
            expert = int(payload.get("expert"))
        except Exception as exc:
            raise MolrCalibrationError(
                f"Validation file has invalid layer/expert values: '{path}'",
            ) from exc

        failure_reasons_raw = payload.get("failure_reasons", [])
        if not isinstance(failure_reasons_raw, list):
            raise MolrCalibrationError(f"Validation file has non-list failure_reasons: '{path}'")

        rows.append(
            {
                "layer": layer,
                "expert": expert,
                "status": str(payload.get("status") or "unknown"),
                "failure_reasons": [str(x) for x in failure_reasons_raw],
                "pred_error_mean": pred_error_mean,
                "true_error_mean": true_error_mean,
                "source_validation_json": str(path),
            }
        )

    if model_seen is None:
        raise MolrCalibrationError("No validation rows were loaded")
    return model_seen, rows


def _build_lookup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_thresholds = sorted({0.0, *[float(r["pred_error_mean"]) for r in rows]})
    table: list[dict[str, Any]] = []
    n_total = len(rows)
    if n_total <= 0:
        raise MolrCalibrationError("Cannot calibrate with zero rows")

    for threshold in unique_thresholds:
        kept = [r for r in rows if float(r["pred_error_mean"]) <= threshold]
        fallback = [r for r in rows if float(r["pred_error_mean"]) > threshold]
        fallback_rate = float(len(fallback)) / float(n_total)
        if kept:
            kept_true_error_mean = float(sum(float(r["true_error_mean"]) for r in kept) / len(kept))
        else:
            kept_true_error_mean = 0.0

        expected_error_proxy = float(sum(float(r["true_error_mean"]) for r in kept) / float(n_total))
        quality_proxy = 1.0 / (1.0 + expected_error_proxy)

        table.append(
            {
                "threshold": float(threshold),
                "fallback_rate": fallback_rate,
                "experts_fallback_total": len(fallback),
                "experts_keep_total": len(kept),
                "kept_true_error_mean": kept_true_error_mean,
                "expected_error_proxy": expected_error_proxy,
                "quality_proxy": quality_proxy,
            }
        )

    return table


def _select_profile_threshold(
    lookup: list[dict[str, Any]],
    quality_target: float,
) -> dict[str, Any]:
    candidates = [row for row in lookup if float(row["quality_proxy"]) >= quality_target]
    if candidates:
        return min(candidates, key=lambda row: float(row["fallback_rate"]))
    return max(lookup, key=lambda row: float(row["quality_proxy"]))


def _cache_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit == 0:
        return []

    ranked = sorted(
        rows,
        key=lambda r: (
            0 if str(r["status"]) != "pass" else 1,
            -float(r["pred_error_mean"]),
            -float(r["true_error_mean"]),
            int(r["layer"]),
            int(r["expert"]),
        ),
    )
    out: list[dict[str, Any]] = []
    for row in ranked[:limit]:
        out.append(
            {
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "priority_score": float(row["pred_error_mean"]),
                "status": str(row["status"]),
                "reason": "validation_failed" if str(row["status"]) != "pass" else "high_predicted_error",
                "failure_reasons": list(row["failure_reasons"]),
            }
        )
    return out


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        checkpoints_dir = Path(args.checkpoints).expanduser().resolve()
        out_json = Path(args.out_json).expanduser().resolve()

        if not checkpoints_dir.is_dir():
            raise MolrCalibrationError(f"--checkpoints path is not a directory: '{checkpoints_dir}'")

        quality_profiles = _parse_quality_profiles(str(args.quality_profiles))
        validation_dir, files = _validation_files(args, checkpoints_dir)
        model_spec, rows = _load_rows(files, expected_model=str(args.model))

        lookup = _build_lookup(rows)
        profiles: dict[str, Any] = {}
        for name, target in sorted(quality_profiles.items()):
            selected = _select_profile_threshold(lookup, target)
            profiles[name] = {
                "quality_proxy_min": float(target),
                "selected_threshold": float(selected["threshold"]),
                "selected_fallback_rate": float(selected["fallback_rate"]),
                "selected_quality_proxy": float(selected["quality_proxy"]),
            }

        payload = {
            "schema_version": MOLR_THRESHOLDS_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": model_spec,
            "calibration_source": {
                "mode": "validation_summary_proxy",
                "checkpoints_dir": str(checkpoints_dir),
                "validation_dir": str(validation_dir),
                "validation_glob": str(args.validation_glob),
                "validation_files_total": len(files),
                "note": (
                    "Threshold quality/fallback sweep uses per-expert validation summary means "
                    "(pred_error_mean/true_error_mean) as a proxy."
                ),
            },
            "metric_definitions": {
                "fallback_decision": "pred_error_mean > threshold",
                "expected_error_proxy": (
                    "Mean true_error_mean over non-fallback experts, zero error assumed for fallback experts."
                ),
                "quality_proxy": "1 / (1 + expected_error_proxy)",
            },
            "summary": {
                "experts_total": len(rows),
                "validation_failed_total": len([r for r in rows if r["status"] != "pass"]),
                "threshold_candidates_total": len(lookup),
            },
            "lookup_table": lookup,
            "quality_profiles": profiles,
            "cache_candidates": _cache_candidates(rows, int(args.top_cache_candidates)),
        }
        _save_json(out_json, payload)

        print(
            "[molr-calibrate] wrote "
            f"experts={len(rows)} thresholds={len(lookup)} "
            f"cache_candidates={len(payload['cache_candidates'])} -> {out_json}",
        )
        return EXIT_OK

    except MolrCalibrationError as exc:
        print(f"[error:molr-calibrate] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
