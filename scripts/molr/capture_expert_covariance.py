#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.types import (
    CHOLESKY_JITTER_SCHEDULE,
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
    MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
    MOLR_ROUTED_TRACES_NPZ_SCHEMA_VERSION,
)


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrCovarianceError(RuntimeError):
    """Raised when covariance capture contract execution fails."""


@dataclass(frozen=True)
class TraceCaptureResult:
    model_spec: str
    d_model: int
    routed_inputs: np.ndarray
    routed_layers: np.ndarray
    routed_experts: np.ndarray


@dataclass(frozen=True)
class SampleSelection:
    source: str
    effective_inputs: np.ndarray
    routed_sample_count: int
    layer_sample_count: int
    fallback_applied: bool


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrCovarianceError(f"Failed writing JSON '{path}': {exc}") from exc


def _save_npz(path: Path, **arrays: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as handle:
            np.savez(handle, **arrays)
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrCovarianceError(f"Failed writing NPZ '{path}': {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 covariance capture utility. Computes per-expert (mu, Cholesky(cov)) "
            "from routed input vectors and emits covariance_stats.npz + summary JSON."
        ),
    )
    parser.add_argument("--model", required=True, help="Model spec associated with this covariance run.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=0,
        help="Nominal token budget for this run; recorded for traceability.",
    )
    parser.add_argument(
        "--routed-inputs-npz",
        default="",
        help=(
            "Optional routed-input contract NPZ path. Expected arrays: "
            "inputs(float[N,D]), layers(int[N]), experts(int[N])."
        ),
    )
    parser.add_argument(
        "--capture-routed-traces",
        action="store_true",
        help="Run integrated routed-trace capture from prompts instead of consuming --routed-inputs-npz.",
    )
    parser.add_argument(
        "--capture-prompts-jsonl",
        default="",
        help=(
            "JSONL source for integrated routed-trace capture mode. "
            "Each non-empty line must be a JSON object containing 'inputs', 'layer', and 'expert'."
        ),
    )
    parser.add_argument(
        "--capture-max-sequences",
        type=int,
        default=0,
        help="Optional max sequences to read in capture mode (0 = no cap).",
    )
    parser.add_argument(
        "--capture-seed",
        type=int,
        default=0,
        help="Deterministic seed for capture-mode sampling decisions.",
    )
    parser.add_argument(
        "--min-samples-per-expert",
        type=int,
        default=16,
        help="Minimum routed samples required to compute covariance per expert. Default: 16.",
    )
    parser.add_argument(
        "--fallback-to-layer-inputs-on-low-samples",
        action="store_true",
        help="When routed samples are low, use layer-wide inputs to estimate covariance for that expert.",
    )
    parser.add_argument(
        "--min-layer-samples-for-fallback",
        type=int,
        default=0,
        help=(
            "Guardrail minimum layer sample count for fallback mode. "
            "0 means use --min-samples-per-expert."
        ),
    )
    parser.add_argument(
        "--max-experts",
        type=int,
        default=0,
        help="Optional cap on number of experts processed after sorting. 0 means no cap.",
    )
    parser.add_argument(
        "--out-routed-traces-npz",
        default="",
        help="Optional output path for routed trace artifact NPZ.",
    )
    parser.add_argument(
        "--trace-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Data type for optional routed trace output and in-memory capture buffers.",
    )
    parser.add_argument(
        "--max-trace-samples-total",
        type=int,
        default=200000,
        help="Upper bound on captured routed rows before dropping additional samples.",
    )
    parser.add_argument(
        "--max-trace-samples-per-expert",
        type=int,
        default=0,
        help="Optional cap per (layer,expert) for captured routed rows (0 = uncapped).",
    )
    parser.add_argument(
        "--max-trace-samples-per-layer",
        type=int,
        default=0,
        help="Optional cap per layer for captured routed rows (0 = uncapped).",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Allow writing empty scaffold artifacts when no routed inputs are provided or capture yields no rows. "
            "For capture mode, this only applies when capture produces zero rows."
        ),
    )
    parser.add_argument("--out-npz", required=True, help="Output path for covariance_stats.npz.")
    parser.add_argument("--out-json", required=True, help="Output path for covariance_summary.json.")

    args = parser.parse_args(argv)
    if args.tokens < 0:
        parser.error("--tokens must be >= 0.")
    if args.capture_max_sequences < 0:
        parser.error("--capture-max-sequences must be >= 0.")
    if args.min_samples_per_expert <= 1:
        parser.error("--min-samples-per-expert must be > 1.")
    if args.min_layer_samples_for_fallback < 0:
        parser.error("--min-layer-samples-for-fallback must be >= 0.")
    if args.max_experts < 0:
        parser.error("--max-experts must be >= 0.")
    if args.max_trace_samples_total <= 0:
        parser.error("--max-trace-samples-total must be > 0.")
    if args.max_trace_samples_per_expert < 0:
        parser.error("--max-trace-samples-per-expert must be >= 0.")
    if args.max_trace_samples_per_layer < 0:
        parser.error("--max-trace-samples-per-layer must be >= 0.")
    if args.capture_routed_traces and args.routed_inputs_npz:
        parser.error("--capture-routed-traces is mutually exclusive with --routed-inputs-npz.")
    if args.capture_routed_traces and not args.capture_prompts_jsonl:
        parser.error("--capture-prompts-jsonl is required when --capture-routed-traces is enabled.")
    if (not args.capture_routed_traces) and args.capture_prompts_jsonl:
        parser.error("--capture-prompts-jsonl requires --capture-routed-traces.")
    if (not args.capture_routed_traces) and args.out_routed_traces_npz:
        parser.error("--out-routed-traces-npz requires --capture-routed-traces.")
    return args


def _as_int(value: Any, *, field: str, line_no: int) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise MolrCovarianceError(f"Line {line_no}: invalid integer '{field}'") from exc
    if out < 0:
        raise MolrCovarianceError(f"Line {line_no}: negative integer '{field}'")
    return out


def _collect_input_vector(raw: Any, *, field: str, line_no: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1:
        raise MolrCovarianceError(f"Line {line_no}: '{field}' must be 1D vector")
    if arr.shape[0] <= 0:
        raise MolrCovarianceError(f"Line {line_no}: '{field}' must be non-empty")
    if not np.isfinite(arr).all():
        raise MolrCovarianceError(f"Line {line_no}: '{field}' contains non-finite values")
    return arr


def _load_routed_contract(path: Path, *, expected_model: str | None) -> TraceCaptureResult:
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise MolrCovarianceError(f"Failed loading routed-inputs NPZ '{path}': {exc}") from exc

    required = ("inputs", "layers", "experts")
    missing = [name for name in required if name not in payload]
    if missing:
        raise MolrCovarianceError(
            f"Routed-input NPZ '{path}' missing required arrays: {', '.join(missing)}",
        )

    inputs = np.asarray(payload["inputs"])
    layers = np.asarray(payload["layers"])
    experts = np.asarray(payload["experts"])

    if inputs.ndim != 2:
        raise MolrCovarianceError(f"inputs must be 2D [N,D], got shape={inputs.shape}")
    if layers.ndim != 1 or experts.ndim != 1:
        raise MolrCovarianceError(
            f"layers/experts must be 1D; got layers={layers.shape}, experts={experts.shape}",
        )
    n_rows = inputs.shape[0]
    if layers.shape[0] != n_rows or experts.shape[0] != n_rows:
        raise MolrCovarianceError(
            "Row count mismatch in routed-input arrays: "
            f"inputs={n_rows}, layers={layers.shape[0]}, experts={experts.shape[0]}",
        )
    if not np.isfinite(inputs).all():
        raise MolrCovarianceError("inputs contain non-finite values.")

    model_spec_raw = payload["model_spec"] if "model_spec" in payload else np.array(expected_model or "")
    model_spec = str(np.asarray(model_spec_raw).item()) if np.asarray(model_spec_raw).shape == () else str(model_spec_raw)
    if expected_model is not None and model_spec and model_spec != expected_model:
        raise MolrCovarianceError(
            f"Model mismatch: routed contract model_spec='{model_spec}' != --model='{expected_model}'",
        )

    return TraceCaptureResult(
        model_spec=(model_spec or (expected_model or "")),
        d_model=int(inputs.shape[1]),
        routed_inputs=inputs.astype(np.float64, copy=False),
        routed_layers=layers.astype(np.int64, copy=False),
        routed_experts=experts.astype(np.int64, copy=False),
    )


def _capture_routed_traces(args: argparse.Namespace) -> tuple[TraceCaptureResult, dict[str, int]]:
    prompts_path = Path(args.capture_prompts_jsonl).expanduser().resolve()
    if not prompts_path.is_file():
        raise MolrCovarianceError(f"--capture-prompts-jsonl path is not a file: '{prompts_path}'")

    per_expert_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    per_layer_count: defaultdict[int, int] = defaultdict(int)

    dropped_by_cap: Counter[str] = Counter()
    inputs_rows: list[np.ndarray] = []
    layers_rows: list[int] = []
    experts_rows: list[int] = []
    d_model: int | None = None

    sequence_rows = prompts_path.read_text(encoding="utf-8").splitlines()
    for seq_index, line in enumerate(sequence_rows):
        if args.capture_max_sequences > 0 and seq_index >= int(args.capture_max_sequences):
            break

        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception as exc:
            raise MolrCovarianceError(f"Line {seq_index + 1}: invalid JSON in capture-prompts JSONL") from exc
        if not isinstance(payload, dict):
            raise MolrCovarianceError(f"Line {seq_index + 1}: JSON record must be an object")

        layer = _as_int(payload.get("layer"), field="layer", line_no=seq_index + 1)
        expert = _as_int(payload.get("expert"), field="expert", line_no=seq_index + 1)
        vec = _collect_input_vector(payload.get("inputs"), field="inputs", line_no=seq_index + 1)
        if d_model is None:
            d_model = int(vec.shape[0])
        elif int(vec.shape[0]) != d_model:
            raise MolrCovarianceError(
                f"Line {seq_index + 1}: d_model mismatch; expected {d_model}, got {int(vec.shape[0])}",
            )

        if len(inputs_rows) >= int(args.max_trace_samples_total):
            dropped_by_cap["max_trace_samples_total"] += 1
            continue
        if args.max_trace_samples_per_expert > 0 and per_expert_count[(layer, expert)] >= int(args.max_trace_samples_per_expert):
            dropped_by_cap["max_trace_samples_per_expert"] += 1
            continue
        if args.max_trace_samples_per_layer > 0 and per_layer_count[layer] >= int(args.max_trace_samples_per_layer):
            dropped_by_cap["max_trace_samples_per_layer"] += 1
            continue

        inputs_rows.append(vec)
        layers_rows.append(layer)
        experts_rows.append(expert)
        per_expert_count[(layer, expert)] += 1
        per_layer_count[layer] += 1

    if not inputs_rows:
        raise MolrCovarianceError("capture_mode_no_rows")

    in_dtype = np.float16 if args.trace_dtype == "float16" else np.float32
    inputs_arr = np.asarray(inputs_rows, dtype=in_dtype).astype(np.float64, copy=False)
    layers_arr = np.asarray(layers_rows, dtype=np.int64)
    experts_arr = np.asarray(experts_rows, dtype=np.int64)

    result = TraceCaptureResult(
        model_spec=str(args.model),
        d_model=int(inputs_arr.shape[1]),
        routed_inputs=inputs_arr,
        routed_layers=layers_arr,
        routed_experts=experts_arr,
    )
    return result, dict(sorted((k, int(v)) for k, v in dropped_by_cap.items()))


def _cholesky_with_jitter(cov: np.ndarray) -> tuple[np.ndarray, float]:
    eye = np.eye(cov.shape[0], dtype=np.float64)
    for jitter in CHOLESKY_JITTER_SCHEDULE:
        try:
            if jitter == 0.0:
                chol = np.linalg.cholesky(cov)
            else:
                chol = np.linalg.cholesky(cov + (jitter * eye))
            return chol, float(jitter)
        except np.linalg.LinAlgError:
            continue
    raise MolrCovarianceError("cholesky_failed_after_jitter_schedule")


def _build_empty_artifacts(
    *,
    args: argparse.Namespace,
    out_npz_path: Path,
    out_json_path: Path,
    reason: str,
    mode: str,
) -> None:
    _save_npz(
        out_npz_path,
        schema_version=np.array(MOLR_COVARIANCE_NPZ_SCHEMA_VERSION),
        model_spec=np.array(args.model),
        d_model=np.array(0, dtype=np.int64),
        layers=np.zeros((0,), dtype=np.int64),
        experts=np.zeros((0,), dtype=np.int64),
        sample_count=np.zeros((0,), dtype=np.int64),
        jitter_used=np.zeros((0,), dtype=np.float64),
        mu=np.zeros((0, 0), dtype=np.float32),
        chol=np.zeros((0, 0, 0), dtype=np.float32),
    )

    summary = {
        "schema_version": MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
        "created_at_utc": _now_utc_iso(),
        "model_spec": args.model,
        "tokens_requested": int(args.tokens),
        "input_contract": {
            "mode": mode,
            "routed_inputs_npz": str(args.routed_inputs_npz) if args.routed_inputs_npz else None,
            "routed_inputs_npz_schema": "molr_routed_inputs.v1",
        },
        "capture_runtime": {
            "status": mode,
        },
        "fallback_policy": {
            "enabled": bool(args.fallback_to_layer_inputs_on_low_samples),
            "min_samples_per_expert": int(args.min_samples_per_expert),
            "min_layer_samples_for_fallback": int(args.min_layer_samples_for_fallback),
            "effective_min_layer_samples_for_fallback": int(
                args.min_layer_samples_for_fallback if args.min_layer_samples_for_fallback > 0 else args.min_samples_per_expert
            ),
        },
        "status": "empty",
        "experts_fallback_used_total": 0,
        "experts_fallback_failed_total": 0,
        "failure_accounting": {
            "global_failures": [{"reason": reason}],
            "by_reason": {reason: 1},
            "experts_failed_total": 0,
            "experts_succeeded_total": 0,
        },
        "experts_succeeded": [],
        "experts_failed": [],
        "outputs": {
            "covariance_stats_npz": str(out_npz_path),
            "covariance_summary_json": str(out_json_path),
        },
    }
    _save_json(out_json_path, summary)


def _build_sample_pools(
    trace: TraceCaptureResult,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, np.ndarray]]:
    expert_indices: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    layer_indices: defaultdict[int, list[int]] = defaultdict(list)

    for idx, (layer, expert) in enumerate(zip(trace.routed_layers.tolist(), trace.routed_experts.tolist())):
        layer_i = int(layer)
        expert_i = int(expert)
        expert_indices[(layer_i, expert_i)].append(idx)
        layer_indices[layer_i].append(idx)

    expert_pool = {
        key: trace.routed_inputs[np.asarray(indices, dtype=np.int64)]
        for key, indices in expert_indices.items()
    }
    layer_pool = {
        layer: trace.routed_inputs[np.asarray(indices, dtype=np.int64)]
        for layer, indices in layer_indices.items()
    }
    return expert_pool, layer_pool


def _select_samples_for_expert(
    *,
    layer: int,
    expert: int,
    expert_pool: dict[tuple[int, int], np.ndarray],
    layer_pool: dict[int, np.ndarray],
    min_samples_per_expert: int,
    fallback_to_layer_inputs_on_low_samples: bool,
    min_layer_samples_for_fallback: int,
) -> SampleSelection:
    routed_x = expert_pool.get((layer, expert), np.zeros((0, 0), dtype=np.float64))
    routed_count = int(routed_x.shape[0])
    layer_x = layer_pool.get(layer, np.zeros((0, 0), dtype=np.float64))
    layer_count = int(layer_x.shape[0])

    if routed_count >= min_samples_per_expert:
        return SampleSelection(
            source="routed",
            effective_inputs=routed_x,
            routed_sample_count=routed_count,
            layer_sample_count=layer_count,
            fallback_applied=False,
        )

    if not fallback_to_layer_inputs_on_low_samples:
        raise MolrCovarianceError(f"insufficient_samples(<{min_samples_per_expert})")

    layer_threshold = min_layer_samples_for_fallback if min_layer_samples_for_fallback > 0 else min_samples_per_expert
    if layer_count < layer_threshold:
        raise MolrCovarianceError(f"insufficient_layer_samples_for_fallback(<{layer_threshold})")

    return SampleSelection(
        source="layer_fallback",
        effective_inputs=layer_x,
        routed_sample_count=routed_count,
        layer_sample_count=layer_count,
        fallback_applied=True,
    )


def _compute_covariance(selection: SampleSelection) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = selection.effective_inputs
    sample_count = int(x.shape[0])
    if sample_count <= 1:
        raise MolrCovarianceError("insufficient_effective_samples(<=1)")
    mu = np.mean(x, axis=0, dtype=np.float64)
    centered = x - mu
    cov = (centered.T @ centered) / float(sample_count - 1)
    chol, jitter_used = _cholesky_with_jitter(cov)
    cov_trace = float(np.trace(cov, dtype=np.float64))
    return mu, chol, float(jitter_used), cov_trace


def _maybe_emit_routed_traces_npz(
    *,
    args: argparse.Namespace,
    trace: TraceCaptureResult,
    out_path: Path | None,
) -> str | None:
    if out_path is None:
        return None

    trace_dtype = np.float16 if args.trace_dtype == "float16" else np.float32
    _save_npz(
        out_path,
        schema_version=np.array(MOLR_ROUTED_TRACES_NPZ_SCHEMA_VERSION),
        model_spec=np.array(trace.model_spec),
        d_model=np.array(int(trace.d_model), dtype=np.int64),
        inputs=trace.routed_inputs.astype(trace_dtype, copy=False),
        layers=trace.routed_layers.astype(np.int64, copy=False),
        experts=trace.routed_experts.astype(np.int64, copy=False),
    )
    return str(out_path)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        out_npz_path = Path(args.out_npz).expanduser().resolve()
        out_json_path = Path(args.out_json).expanduser().resolve()
        out_routed_traces_path = (
            Path(args.out_routed_traces_npz).expanduser().resolve()
            if args.out_routed_traces_npz
            else None
        )

        capture_mode = bool(args.capture_routed_traces)
        capture_runtime_status = "capture_enabled" if capture_mode else "contract_only"
        capture_dropped_counters: dict[str, int] = {}

        if capture_mode:
            try:
                trace_result, capture_dropped_counters = _capture_routed_traces(args)
            except MolrCovarianceError as exc:
                if str(exc) == "capture_mode_no_rows" and args.allow_empty:
                    _build_empty_artifacts(
                        args=args,
                        out_npz_path=out_npz_path,
                        out_json_path=out_json_path,
                        reason="capture_mode_no_rows",
                        mode=capture_runtime_status,
                    )
                    print(f"[molr-cov] wrote empty scaffold artifacts -> {out_npz_path} and {out_json_path}")
                    return EXIT_OK
                raise
        else:
            if not args.routed_inputs_npz:
                if not args.allow_empty:
                    raise MolrCovarianceError(
                        "No --routed-inputs-npz provided. Enable --capture-routed-traces with "
                        "--capture-prompts-jsonl or pass --allow-empty.",
                    )
                _build_empty_artifacts(
                    args=args,
                    out_npz_path=out_npz_path,
                    out_json_path=out_json_path,
                    reason="missing_routed_inputs_npz",
                    mode=capture_runtime_status,
                )
                print(f"[molr-cov] wrote empty scaffold artifacts -> {out_npz_path} and {out_json_path}")
                return EXIT_OK

            routed_path = Path(args.routed_inputs_npz).expanduser().resolve()
            if not routed_path.is_file():
                raise MolrCovarianceError(f"--routed-inputs-npz path is not a file: '{routed_path}'")
            trace_result = _load_routed_contract(routed_path, expected_model=str(args.model))

        if trace_result.routed_inputs.shape[0] == 0:
            if args.allow_empty:
                _build_empty_artifacts(
                    args=args,
                    out_npz_path=out_npz_path,
                    out_json_path=out_json_path,
                    reason="no_rows_after_capture_or_load",
                    mode=capture_runtime_status,
                )
                print(f"[molr-cov] wrote empty scaffold artifacts -> {out_npz_path} and {out_json_path}")
                return EXIT_OK
            raise MolrCovarianceError("No routed rows available for covariance computation.")

        if not np.isfinite(trace_result.routed_inputs).all():
            raise MolrCovarianceError("inputs contain non-finite values.")

        d_model = int(trace_result.d_model)
        if d_model <= 0:
            raise MolrCovarianceError("d_model must be > 0")

        expert_pool, layer_pool = _build_sample_pools(trace_result)

        unique_keys_all = sorted(expert_pool.keys())
        unique_keys = unique_keys_all if args.max_experts <= 0 else unique_keys_all[: int(args.max_experts)]

        success_layers: list[int] = []
        success_experts: list[int] = []
        success_counts: list[int] = []
        success_jitter: list[float] = []
        success_mu: list[np.ndarray] = []
        success_chol: list[np.ndarray] = []

        experts_succeeded: list[dict[str, Any]] = []
        experts_failed: list[dict[str, Any]] = []
        failure_reasons: Counter[str] = Counter()
        fallback_used_total = 0
        fallback_failed_total = 0

        for layer, expert in unique_keys:
            try:
                selection = _select_samples_for_expert(
                    layer=layer,
                    expert=expert,
                    expert_pool=expert_pool,
                    layer_pool=layer_pool,
                    min_samples_per_expert=int(args.min_samples_per_expert),
                    fallback_to_layer_inputs_on_low_samples=bool(args.fallback_to_layer_inputs_on_low_samples),
                    min_layer_samples_for_fallback=int(args.min_layer_samples_for_fallback),
                )
            except MolrCovarianceError as exc:
                reason = str(exc)
                routed_count = int(expert_pool.get((layer, expert), np.zeros((0, 0), dtype=np.float64)).shape[0])
                layer_count = int(layer_pool.get(layer, np.zeros((0, 0), dtype=np.float64)).shape[0])
                if bool(args.fallback_to_layer_inputs_on_low_samples) and routed_count < int(args.min_samples_per_expert):
                    fallback_failed_total += 1
                experts_failed.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "sample_count": routed_count,
                        "routed_sample_count": routed_count,
                        "layer_sample_count": layer_count,
                        "fallback_applied": False,
                        "reason": reason,
                    }
                )
                failure_reasons[reason] += 1
                continue

            try:
                mu, chol, jitter_used, cov_trace = _compute_covariance(selection)
            except MolrCovarianceError as exc:
                reason = str(exc)
                experts_failed.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "sample_count": int(selection.effective_inputs.shape[0]),
                        "routed_sample_count": selection.routed_sample_count,
                        "layer_sample_count": selection.layer_sample_count,
                        "fallback_applied": bool(selection.fallback_applied),
                        "reason": reason,
                    }
                )
                failure_reasons[reason] += 1
                continue

            effective_sample_count = int(selection.effective_inputs.shape[0])
            success_layers.append(layer)
            success_experts.append(expert)
            success_counts.append(effective_sample_count)
            success_jitter.append(float(jitter_used))
            success_mu.append(mu.astype(np.float32, copy=False))
            success_chol.append(chol.astype(np.float32, copy=False))

            if selection.fallback_applied:
                fallback_used_total += 1

            experts_succeeded.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "sample_count": effective_sample_count,
                    "sample_source": selection.source,
                    "routed_sample_count": selection.routed_sample_count,
                    "layer_sample_count": selection.layer_sample_count,
                    "effective_sample_count": effective_sample_count,
                    "fallback_applied": bool(selection.fallback_applied),
                    "d_model": d_model,
                    "jitter_used": float(jitter_used),
                    "cov_trace": float(cov_trace),
                }
            )

        if success_mu:
            mu_arr = np.stack(success_mu, axis=0)
            chol_arr = np.stack(success_chol, axis=0)
        else:
            mu_arr = np.zeros((0, d_model), dtype=np.float32)
            chol_arr = np.zeros((0, d_model, d_model), dtype=np.float32)

        _save_npz(
            out_npz_path,
            schema_version=np.array(MOLR_COVARIANCE_NPZ_SCHEMA_VERSION),
            model_spec=np.array(trace_result.model_spec or str(args.model)),
            d_model=np.array(d_model, dtype=np.int64),
            layers=np.asarray(success_layers, dtype=np.int64),
            experts=np.asarray(success_experts, dtype=np.int64),
            sample_count=np.asarray(success_counts, dtype=np.int64),
            jitter_used=np.asarray(success_jitter, dtype=np.float64),
            mu=mu_arr,
            chol=chol_arr,
        )

        routed_traces_path_str = _maybe_emit_routed_traces_npz(
            args=args,
            trace=trace_result,
            out_path=out_routed_traces_path,
        )

        effective_layer_threshold = (
            int(args.min_layer_samples_for_fallback)
            if int(args.min_layer_samples_for_fallback) > 0
            else int(args.min_samples_per_expert)
        )

        input_contract: dict[str, Any] = {
            "mode": capture_runtime_status,
            "routed_inputs_npz_schema": "molr_routed_inputs.v1",
        }
        if capture_mode:
            input_contract["capture_prompts_jsonl"] = str(Path(args.capture_prompts_jsonl).expanduser().resolve())
            input_contract["capture_row_schema"] = {
                "required_fields": ["inputs", "layer", "expert"],
            }
        else:
            input_contract["routed_inputs_npz"] = str(Path(args.routed_inputs_npz).expanduser().resolve())
            input_contract["required_arrays"] = ["inputs", "layers", "experts"]

        summary = {
            "schema_version": MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": trace_result.model_spec or str(args.model),
            "tokens_requested": int(args.tokens),
            "input_contract": input_contract,
            "capture_runtime": {
                "status": capture_runtime_status,
                "trace_dtype": str(args.trace_dtype),
                "capture_max_sequences": int(args.capture_max_sequences),
                "capture_seed": int(args.capture_seed),
                "max_trace_samples_total": int(args.max_trace_samples_total),
                "max_trace_samples_per_expert": int(args.max_trace_samples_per_expert),
                "max_trace_samples_per_layer": int(args.max_trace_samples_per_layer),
                "dropped_samples": capture_dropped_counters,
            },
            "config": {
                "min_samples_per_expert": int(args.min_samples_per_expert),
                "max_experts": int(args.max_experts),
                "cholesky_jitter_schedule": list(CHOLESKY_JITTER_SCHEDULE),
            },
            "fallback_policy": {
                "enabled": bool(args.fallback_to_layer_inputs_on_low_samples),
                "min_samples_per_expert": int(args.min_samples_per_expert),
                "min_layer_samples_for_fallback": int(args.min_layer_samples_for_fallback),
                "effective_min_layer_samples_for_fallback": effective_layer_threshold,
            },
            "observed": {
                "rows_total": int(trace_result.routed_inputs.shape[0]),
                "d_model": d_model,
                "layers_observed_total": len(layer_pool),
                "experts_observed_total": len(unique_keys_all),
                "experts_processed_total": len(unique_keys),
            },
            "experts_fallback_used_total": int(fallback_used_total),
            "experts_fallback_failed_total": int(fallback_failed_total),
            "failure_accounting": {
                "by_reason": dict(sorted(failure_reasons.items())),
                "experts_failed_total": len(experts_failed),
                "experts_succeeded_total": len(experts_succeeded),
            },
            "experts_succeeded": experts_succeeded,
            "experts_failed": experts_failed,
            "outputs": {
                "covariance_stats_npz": str(out_npz_path),
                "covariance_summary_json": str(out_json_path),
                "covariance_npz_schema_version": MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
                "routed_traces_npz": routed_traces_path_str,
                "routed_traces_npz_schema_version": (
                    MOLR_ROUTED_TRACES_NPZ_SCHEMA_VERSION if routed_traces_path_str is not None else None
                ),
            },
        }
        _save_json(out_json_path, summary)

        print(
            "[molr-cov] wrote "
            f"success={len(experts_succeeded)} failed={len(experts_failed)} "
            f"d_model={d_model} -> {out_npz_path}",
        )
        return EXIT_OK

    except MolrCovarianceError as exc:
        print(f"[error:molr-cov] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
