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

from molr.runtime_bundle import MolrRuntimeBundleError, load_runtime_bundle
from molr.types import MOLR_RUNTIME_SHADOW_REPORT_SCHEMA_VERSION, MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrRuntimeShadowError(RuntimeError):
    """Raised when runtime shadow validation cannot proceed."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrRuntimeShadowError(f"Failed reading JSON '{path}': {exc}") from exc


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrRuntimeShadowError(f"Failed writing JSON '{path}': {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-4 guarded runtime shadow harness. Validates bundle/config contracts and "
            "summarizes telemetry counters without changing default inference behavior."
        ),
    )
    parser.add_argument("--bundle-dir", required=True, help="Path to packaged MoLR bundle directory.")
    parser.add_argument(
        "--runtime-config-json",
        required=True,
        help="Runtime opt-in configuration JSON (enabled, threshold/profile, telemetry toggles).",
    )
    parser.add_argument(
        "--telemetry-json",
        default="",
        help=(
            "Optional telemetry snapshot JSON from shadow runs. "
            "If omitted, report contains contract validation and empty counters."
        ),
    )
    parser.add_argument("--model", default="", help="Optional model spec guard against bundle manifest.")
    parser.add_argument(
        "--fallback-rate-alert-threshold",
        type=float,
        default=0.25,
        help="Fallback-rate alert threshold in [0,1]. Default: 0.25.",
    )
    parser.add_argument(
        "--latency-ratio-alert-threshold",
        type=float,
        default=1.50,
        help="Alert when avg fallback latency / avg molr latency exceeds this ratio. Default: 1.50.",
    )
    parser.add_argument(
        "--require-explicit-enable",
        action="store_true",
        help="Require runtime config to set enabled=true (recommended for guarded rollout).",
    )
    parser.add_argument("--out-json", required=True, help="Output path for runtime shadow report JSON.")

    args = parser.parse_args(argv)
    if args.fallback_rate_alert_threshold < 0.0 or args.fallback_rate_alert_threshold > 1.0:
        parser.error("--fallback-rate-alert-threshold must be in [0,1]")
    if args.latency_ratio_alert_threshold <= 0.0:
        parser.error("--latency-ratio-alert-threshold must be > 0")
    return args


def _load_telemetry(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_json(path)
    schema = str(payload.get("schema_version") or "")
    if schema != MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION:
        raise MolrRuntimeShadowError(
            f"Telemetry schema mismatch: got '{schema}', expected '{MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION}'",
        )
    rows = payload.get("experts", [])
    if not isinstance(rows, list):
        raise MolrRuntimeShadowError("Telemetry experts must be a list")
    return rows, payload


def _to_int(value: Any, *, field: str, context: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise MolrRuntimeShadowError(f"Invalid integer for {field} in {context}") from exc
    if out < 0:
        raise MolrRuntimeShadowError(f"Negative integer for {field} in {context}: {out}")
    return out


def _to_float(value: Any, *, field: str, context: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise MolrRuntimeShadowError(f"Invalid float for {field} in {context}") from exc
    if not math.isfinite(out):
        raise MolrRuntimeShadowError(f"Non-finite float for {field} in {context}")
    return out


def _summarize_telemetry(
    rows: list[dict[str, Any]],
    *,
    fallback_rate_alert_threshold: float,
    latency_ratio_alert_threshold: float,
) -> dict[str, Any]:
    totals = {
        "calls_total": 0,
        "fallback_calls_total": 0,
        "fallback_rate": 0.0,
    }

    expert_rows: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    sum_pred_error = 0.0
    pred_error_weight = 0
    sum_molr_ms = 0.0
    sum_fallback_ms = 0.0
    latency_weight_molr = 0
    latency_weight_fallback = 0

    for raw in rows:
        if not isinstance(raw, dict):
            raise MolrRuntimeShadowError("Telemetry expert row must be object")
        layer = _to_int(raw.get("layer"), field="layer", context="telemetry row")
        expert = _to_int(raw.get("expert"), field="expert", context="telemetry row")
        calls = _to_int(raw.get("calls_total", 0), field="calls_total", context=f"expert ({layer},{expert})")
        fallback_calls = _to_int(
            raw.get("fallback_calls_total", 0),
            field="fallback_calls_total",
            context=f"expert ({layer},{expert})",
        )
        if fallback_calls > calls:
            raise MolrRuntimeShadowError(
                f"Invalid telemetry for expert ({layer},{expert}): fallback_calls_total > calls_total",
            )

        pred_error_mean = _to_float(
            raw.get("predicted_error_mean", 0.0),
            field="predicted_error_mean",
            context=f"expert ({layer},{expert})",
        )
        avg_molr_latency_ms = _to_float(
            raw.get("avg_molr_latency_ms", 0.0),
            field="avg_molr_latency_ms",
            context=f"expert ({layer},{expert})",
        )
        avg_fallback_latency_ms = _to_float(
            raw.get("avg_fallback_latency_ms", 0.0),
            field="avg_fallback_latency_ms",
            context=f"expert ({layer},{expert})",
        )

        fallback_rate = float(fallback_calls) / float(calls) if calls > 0 else 0.0
        latency_ratio = (
            (avg_fallback_latency_ms / avg_molr_latency_ms)
            if avg_molr_latency_ms > 0.0 and avg_fallback_latency_ms > 0.0
            else None
        )

        totals["calls_total"] += calls
        totals["fallback_calls_total"] += fallback_calls
        if calls > 0:
            pred_error_weight += calls
            sum_pred_error += pred_error_mean * calls
        if avg_molr_latency_ms > 0.0 and calls > 0:
            sum_molr_ms += avg_molr_latency_ms * calls
            latency_weight_molr += calls
        if avg_fallback_latency_ms > 0.0 and fallback_calls > 0:
            sum_fallback_ms += avg_fallback_latency_ms * fallback_calls
            latency_weight_fallback += fallback_calls

        expert_row = {
            "layer": layer,
            "expert": expert,
            "calls_total": calls,
            "fallback_calls_total": fallback_calls,
            "fallback_rate": fallback_rate,
            "predicted_error_mean": pred_error_mean,
            "avg_molr_latency_ms": avg_molr_latency_ms,
            "avg_fallback_latency_ms": avg_fallback_latency_ms,
            "fallback_latency_ratio": latency_ratio,
        }
        expert_rows.append(expert_row)

        if fallback_rate > fallback_rate_alert_threshold:
            alerts.append(
                {
                    "kind": "high_fallback_rate",
                    "layer": layer,
                    "expert": expert,
                    "observed": fallback_rate,
                    "threshold": fallback_rate_alert_threshold,
                }
            )
        if latency_ratio is not None and latency_ratio > latency_ratio_alert_threshold:
            alerts.append(
                {
                    "kind": "high_latency_ratio",
                    "layer": layer,
                    "expert": expert,
                    "observed": latency_ratio,
                    "threshold": latency_ratio_alert_threshold,
                }
            )

    if totals["calls_total"] > 0:
        totals["fallback_rate"] = float(totals["fallback_calls_total"]) / float(totals["calls_total"])

    aggregate = {
        "predicted_error_mean_weighted": (
            sum_pred_error / float(pred_error_weight) if pred_error_weight > 0 else 0.0
        ),
        "avg_molr_latency_ms_weighted": (
            sum_molr_ms / float(latency_weight_molr) if latency_weight_molr > 0 else 0.0
        ),
        "avg_fallback_latency_ms_weighted": (
            sum_fallback_ms / float(latency_weight_fallback) if latency_weight_fallback > 0 else 0.0
        ),
    }
    if aggregate["avg_molr_latency_ms_weighted"] > 0.0 and aggregate["avg_fallback_latency_ms_weighted"] > 0.0:
        aggregate["fallback_over_molr_latency_ratio_weighted"] = (
            aggregate["avg_fallback_latency_ms_weighted"] / aggregate["avg_molr_latency_ms_weighted"]
        )
    else:
        aggregate["fallback_over_molr_latency_ratio_weighted"] = None

    expert_rows.sort(key=lambda row: (int(row["layer"]), int(row["expert"])))
    return {
        "totals": totals,
        "aggregate": aggregate,
        "experts": expert_rows,
        "alerts": alerts,
    }


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        runtime_cfg = Path(args.runtime_config_json).expanduser().resolve()
        out_json = Path(args.out_json).expanduser().resolve()

        try:
            runtime_state = load_runtime_bundle(
                bundle_dir=bundle_dir,
                runtime_config_path=runtime_cfg,
                expected_model=str(args.model) or None,
                require_explicit_enable=bool(args.require_explicit_enable),
            )
        except MolrRuntimeBundleError as exc:
            raise MolrRuntimeShadowError(str(exc)) from exc

        telemetry_payload: dict[str, Any] = {}
        telemetry_rows: list[dict[str, Any]] = []
        if str(args.telemetry_json).strip():
            telemetry_path = Path(args.telemetry_json).expanduser().resolve()
            if not telemetry_path.is_file():
                raise MolrRuntimeShadowError(f"--telemetry-json path is not a file: '{telemetry_path}'")
            telemetry_rows, telemetry_payload = _load_telemetry(telemetry_path)

        telemetry_summary = _summarize_telemetry(
            telemetry_rows,
            fallback_rate_alert_threshold=float(args.fallback_rate_alert_threshold),
            latency_ratio_alert_threshold=float(args.latency_ratio_alert_threshold),
        )

        report = {
            "schema_version": MOLR_RUNTIME_SHADOW_REPORT_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "mode": "shadow_validation",
            "model_spec": runtime_state["model_spec"],
            "runtime_contract": {
                "bundle_dir": runtime_state["bundle_dir"],
                "manifest_path": runtime_state["manifest_path"],
                "thresholds_path": runtime_state["thresholds_path"],
                "plan_path": runtime_state["plan_path"],
                "runtime_config_path": runtime_state["runtime_config_path"],
                "coverage": runtime_state.get("coverage", {}),
            },
            "runtime": runtime_state["runtime"],
            "telemetry_input": {
                "schema_version": telemetry_payload.get("schema_version", ""),
                "source": telemetry_payload.get("source", ""),
                "window": telemetry_payload.get("window", {}),
                "experts_total": len(telemetry_rows),
            },
            "telemetry_summary": telemetry_summary,
            "safety": {
                "default_inference_behavior_unchanged": True,
                "requires_explicit_opt_in": bool(args.require_explicit_enable),
                "auto_disable_recommendations": telemetry_summary["alerts"],
            },
        }
        _save_json(out_json, report)

        print(
            "[molr-runtime-shadow] wrote "
            f"experts={len(telemetry_rows)} alerts={len(telemetry_summary['alerts'])} -> {out_json}",
        )
        return EXIT_OK

    except MolrRuntimeShadowError as exc:
        print(f"[error:molr-runtime-shadow] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
