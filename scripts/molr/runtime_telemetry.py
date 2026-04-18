#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.types import MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrRuntimeTelemetryError(RuntimeError):
    """Raised when runtime telemetry aggregation fails."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrRuntimeTelemetryError(f"Failed writing JSON '{path}': {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-4 runtime telemetry aggregator for MoLR fallback/latency counters. "
            "Consumes JSONL expert-call events and emits schema-stable telemetry snapshot."
        ),
    )
    parser.add_argument(
        "--events-jsonl",
        required=True,
        help=(
            "Path to JSONL file with one expert-call event per line: "
            "{layer, expert, used_fallback, predicted_error, molr_latency_ms, fallback_latency_ms}."
        ),
    )
    parser.add_argument("--model", required=True, help="Model spec recorded in telemetry artifact.")
    parser.add_argument(
        "--source",
        default="runtime_events_jsonl",
        help="Source label included in telemetry metadata.",
    )
    parser.add_argument("--out-json", required=True, help="Output path for telemetry snapshot JSON.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero valid events and emit empty telemetry snapshot.",
    )
    return parser.parse_args(argv)


def _to_int(value: Any, *, field: str, line_no: int) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise MolrRuntimeTelemetryError(f"Line {line_no}: invalid integer field '{field}'") from exc
    if out < 0:
        raise MolrRuntimeTelemetryError(f"Line {line_no}: negative integer field '{field}'")
    return out


def _to_float(value: Any, *, field: str, line_no: int) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise MolrRuntimeTelemetryError(f"Line {line_no}: invalid float field '{field}'") from exc
    if not math.isfinite(out):
        raise MolrRuntimeTelemetryError(f"Line {line_no}: non-finite float field '{field}'")
    return out


def _to_bool(value: Any, *, field: str, line_no: int) -> bool:
    if isinstance(value, bool):
        return value
    raise MolrRuntimeTelemetryError(f"Line {line_no}: invalid bool field '{field}'")


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MolrRuntimeTelemetryError(f"--events-jsonl path is not a file: '{path}'")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception as exc:
            raise MolrRuntimeTelemetryError(f"Line {line_no}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise MolrRuntimeTelemetryError(f"Line {line_no}: event must be JSON object")
        row = {
            "layer": _to_int(payload.get("layer"), field="layer", line_no=line_no),
            "expert": _to_int(payload.get("expert"), field="expert", line_no=line_no),
            "used_fallback": _to_bool(payload.get("used_fallback"), field="used_fallback", line_no=line_no),
            "predicted_error": _to_float(
                payload.get("predicted_error", 0.0),
                field="predicted_error",
                line_no=line_no,
            ),
            "molr_latency_ms": _to_float(
                payload.get("molr_latency_ms", 0.0),
                field="molr_latency_ms",
                line_no=line_no,
            ),
            "fallback_latency_ms": _to_float(
                payload.get("fallback_latency_ms", 0.0),
                field="fallback_latency_ms",
                line_no=line_no,
            ),
        }
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acc: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {
            "calls_total": 0,
            "fallback_calls_total": 0,
            "pred_error_sum": 0.0,
            "molr_latency_sum": 0.0,
            "fallback_latency_sum": 0.0,
        }
    )

    for row in rows:
        key = (int(row["layer"]), int(row["expert"]))
        item = acc[key]
        item["calls_total"] += 1
        if bool(row["used_fallback"]):
            item["fallback_calls_total"] += 1
        item["pred_error_sum"] += float(row["predicted_error"])
        item["molr_latency_sum"] += float(row["molr_latency_ms"])
        item["fallback_latency_sum"] += float(row["fallback_latency_ms"])

    out: list[dict[str, Any]] = []
    for (layer, expert), item in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        calls = int(item["calls_total"])
        fallback_calls = int(item["fallback_calls_total"])
        out.append(
            {
                "layer": layer,
                "expert": expert,
                "calls_total": calls,
                "fallback_calls_total": fallback_calls,
                "predicted_error_mean": (float(item["pred_error_sum"]) / float(calls)) if calls > 0 else 0.0,
                "avg_molr_latency_ms": (float(item["molr_latency_sum"]) / float(calls)) if calls > 0 else 0.0,
                "avg_fallback_latency_ms": (
                    float(item["fallback_latency_sum"]) / float(fallback_calls)
                    if fallback_calls > 0
                    else 0.0
                ),
            }
        )
    return out


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        events_path = Path(args.events_jsonl).expanduser().resolve()
        out_path = Path(args.out_json).expanduser().resolve()
        rows = _load_events(events_path)
        if not rows and not args.allow_empty:
            raise MolrRuntimeTelemetryError("No valid events were parsed; use --allow-empty to emit scaffold")

        experts = _aggregate(rows)
        payload = {
            "schema_version": MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": str(args.model),
            "source": str(args.source),
            "window": {
                "events_total": len(rows),
                "experts_total": len(experts),
            },
            "metric_definitions": {
                "fallback_rate": "fallback_calls_total / calls_total",
                "predicted_error_mean": "mean(predicted_error)",
                "avg_molr_latency_ms": "mean(molr_latency_ms)",
                "avg_fallback_latency_ms": "mean(fallback_latency_ms where used_fallback=true)",
            },
            "experts": experts,
        }
        _save_json(out_path, payload)

        print(
            "[molr-runtime-telemetry] wrote "
            f"events={len(rows)} experts={len(experts)} -> {out_path}",
        )
        return EXIT_OK
    except MolrRuntimeTelemetryError as exc:
        print(f"[error:molr-runtime-telemetry] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
