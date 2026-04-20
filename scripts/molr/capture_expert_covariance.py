#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2,
    MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
    MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION_V2,
    MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION,
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


@dataclass(frozen=True)
class PromptInferenceRecord:
    prompt: str
    inference_params: dict[str, Any]


@dataclass(frozen=True)
class PromptInferenceSpec:
    records: list[PromptInferenceRecord]
    source_schema: str


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
        "--input-granularity",
        choices=("layer", "expert", "auto"),
        default="auto",
        help="Input aggregation granularity for covariance fitting. 'auto' defaults to layer mode.",
    )
    parser.add_argument(
        "--capture-layer-traces",
        action="store_true",
        help="Run integrated layer-input trace capture from prompts instead of consuming --routed-inputs-npz.",
    )
    parser.add_argument(
        "--capture-routed-traces",
        action="store_true",
        help="Deprecated alias for --capture-layer-traces.",
    )
    parser.add_argument(
        "--capture-prompts-jsonl",
        default="",
        help=(
            "JSON/JSONL prompt+inference source for integrated routed-trace capture mode. "
            "Accepts either a JSON object with a top-level 'records' array or JSONL with one record per line. "
            "Each record requires 'prompt' (string) and optional 'inference_params' (object)."
        ),
    )
    parser.add_argument(
        "--capture-trace-jsonl",
        default="",
        help=(
            "Optional override path for routed trace JSONL. If unset, capture mode will use the value in "
            "LLAMA_MOE_TRACE_JSONL from the environment."
        ),
    )
    parser.add_argument(
        "--capture-llama-cli",
        default="",
        help=(
            "Optional llama CLI binary path for integrated inference capture bridge. "
            "Defaults to env LLAMA_MOLR_CLI_BIN when set, otherwise 'llama-cli'."
        ),
    )
    parser.add_argument(
        "--capture-common-inference-params",
        default="",
        help=(
            "Optional JSON object string merged into each record's inference_params "
            "(record-level keys take precedence)."
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
        help="Minimum samples required per expert (expert mode) or compatibility alias for layer mode.",
    )
    parser.add_argument(
        "--min-samples-per-layer",
        type=int,
        default=0,
        help="Minimum samples required per layer in layer mode. 0 means use --min-samples-per-expert.",
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
        help="Optional output path for routed trace artifact NPZ (expert mode only).",
    )
    parser.add_argument(
        "--out-layer-traces-npz",
        default="",
        help="Optional output path for layer trace artifact NPZ.",
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
    if args.min_samples_per_layer < 0:
        parser.error("--min-samples-per-layer must be >= 0.")

    capture_routed_alias_used = bool(args.capture_routed_traces)
    capture_enabled = bool(args.capture_layer_traces or capture_routed_alias_used)

    if capture_enabled and args.routed_inputs_npz:
        parser.error("--capture-layer-traces/--capture-routed-traces is mutually exclusive with --routed-inputs-npz.")
    if capture_enabled and not args.capture_prompts_jsonl:
        parser.error("--capture-prompts-jsonl is required when capture mode is enabled.")
    if (not capture_enabled) and args.capture_prompts_jsonl:
        parser.error("--capture-prompts-jsonl requires --capture-layer-traces.")
    if (not capture_enabled) and args.capture_trace_jsonl:
        parser.error("--capture-trace-jsonl requires --capture-layer-traces.")
    if (not capture_enabled) and args.capture_llama_cli:
        parser.error("--capture-llama-cli requires --capture-layer-traces.")
    if (not capture_enabled) and args.capture_common_inference_params:
        parser.error("--capture-common-inference-params requires --capture-layer-traces.")
    if (not capture_enabled) and args.out_routed_traces_npz:
        parser.error("--out-routed-traces-npz requires capture mode.")
    if (not capture_enabled) and args.out_layer_traces_npz:
        parser.error("--out-layer-traces-npz requires capture mode.")

    if args.input_granularity == "layer" and args.out_routed_traces_npz:
        parser.error("--out-routed-traces-npz is not valid when --input-granularity=layer.")

    args.capture_layer_traces = capture_enabled
    args.capture_routed_traces = capture_enabled

    if args.input_granularity == "auto":
        args.input_granularity_resolved = "layer"
    else:
        args.input_granularity_resolved = args.input_granularity

    args.min_samples_per_layer_effective = (
        int(args.min_samples_per_layer)
        if int(args.min_samples_per_layer) > 0
        else int(args.min_samples_per_expert)
    )

    args.deprecation_warnings = []
    if capture_routed_alias_used:
        args.deprecation_warnings.append("--capture-routed-traces is deprecated; use --capture-layer-traces")

    if args.input_granularity_resolved == "layer":
        if args.fallback_to_layer_inputs_on_low_samples:
            args.deprecation_warnings.append(
                "--fallback-to-layer-inputs-on-low-samples is ignored in layer mode",
            )
        if int(args.min_layer_samples_for_fallback) > 0:
            args.deprecation_warnings.append(
                "--min-layer-samples-for-fallback is ignored in layer mode",
            )

    if args.capture_common_inference_params:
        try:
            common_payload = json.loads(args.capture_common_inference_params)
        except Exception as exc:
            parser.error(f"--capture-common-inference-params must be valid JSON object: {exc}")
        if not isinstance(common_payload, dict):
            parser.error("--capture-common-inference-params must decode to a JSON object.")
    else:
        common_payload = {}
    args.capture_common_inference_params_obj = dict(common_payload)
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


def _parse_prompt_record(raw: Any, *, line_no: int) -> PromptInferenceRecord:
    if not isinstance(raw, dict):
        raise MolrCovarianceError(f"Line {line_no}: capture prompt record must be a JSON object")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise MolrCovarianceError(f"Line {line_no}: 'prompt' must be a non-empty string")

    inference_params = raw.get("inference_params", {})
    if inference_params is None:
        inference_params = {}
    if not isinstance(inference_params, dict):
        raise MolrCovarianceError(f"Line {line_no}: 'inference_params' must be a JSON object when provided")

    return PromptInferenceRecord(prompt=prompt, inference_params=dict(inference_params))


def _load_capture_prompt_inference_spec(args: argparse.Namespace) -> PromptInferenceSpec:
    prompts_path = Path(args.capture_prompts_jsonl).expanduser().resolve()
    if not prompts_path.is_file():
        raise MolrCovarianceError(f"--capture-prompts-jsonl path is not a file: '{prompts_path}'")

    raw_text = prompts_path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        raise MolrCovarianceError("capture prompt/inference file is empty")

    records: list[PromptInferenceRecord] = []
    source_schema = ""

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            if "records" in payload:
                source_schema = str(payload.get("schema", "capture_prompt_inference.v1"))
                raw_records = payload.get("records")
                if not isinstance(raw_records, list) or len(raw_records) == 0:
                    raise MolrCovarianceError("JSON capture file requires non-empty top-level 'records' array")
                for idx, raw_record in enumerate(raw_records, start=1):
                    records.append(_parse_prompt_record(raw_record, line_no=idx))
            elif "prompt" in payload:
                source_schema = "capture_prompt_inference.v1"
                records.append(_parse_prompt_record(payload, line_no=1))
            else:
                raise MolrCovarianceError(
                    "JSON capture file object must contain either 'records' or 'prompt'",
                )
        elif isinstance(payload, list):
            source_schema = "capture_prompt_inference.v1"
            if len(payload) == 0:
                raise MolrCovarianceError("JSON capture file array is empty")
            for idx, raw_record in enumerate(payload, start=1):
                records.append(_parse_prompt_record(raw_record, line_no=idx))
        else:
            raise MolrCovarianceError("capture prompt/inference JSON must be object or array")
    except MolrCovarianceError:
        raise
    except Exception:
        # Fallback to JSONL parsing.
        source_schema = "capture_prompt_inference_jsonl.v1"
        lines = raw_text.splitlines()
        for idx, line in enumerate(lines, start=1):
            token = line.strip()
            if not token:
                continue
            try:
                raw_record = json.loads(token)
            except Exception as exc:
                raise MolrCovarianceError(f"Line {idx}: invalid JSON in capture prompt/inference JSONL") from exc
            records.append(_parse_prompt_record(raw_record, line_no=idx))

    if args.capture_max_sequences > 0:
        records = records[: int(args.capture_max_sequences)]
    if len(records) == 0:
        raise MolrCovarianceError("capture prompt/inference file has no usable records")

    common_params = dict(getattr(args, "capture_common_inference_params_obj", {}))
    if common_params:
        merged_records: list[PromptInferenceRecord] = []
        for record in records:
            merged = dict(common_params)
            merged.update(record.inference_params)
            merged_records.append(PromptInferenceRecord(prompt=record.prompt, inference_params=merged))
        records = merged_records

    return PromptInferenceSpec(records=records, source_schema=source_schema)


def _as_number(value: Any, *, field: str, line_no: int) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise MolrCovarianceError(f"Line {line_no}: invalid numeric '{field}'") from exc
    if not np.isfinite(out):
        raise MolrCovarianceError(f"Line {line_no}: non-finite numeric '{field}'")
    return out


def _extract_trace_fields(
    raw: Any,
    *,
    line_no: int,
    input_granularity: str,
) -> tuple[np.ndarray, int, int | None] | None:
    if not isinstance(raw, dict):
        return None

    event = raw.get("event")
    accepted_events = {
        "moe_routed_input",
        "routed_input",
        "moe.trace.routed_input",
        "moe_layer_input",
    }
    if isinstance(event, str) and event not in accepted_events:
        # Ignore unrelated events when a multi-event trace file is used.
        return None

    node: Any = raw
    if isinstance(raw.get("data"), dict):
        node = raw["data"]

    if not isinstance(node, dict):
        return None

    layer_raw = node.get("layer", node.get("layer_id"))
    expert_raw = node.get("expert", node.get("expert_id"))
    inputs_raw = node.get("inputs", node.get("input", node.get("vector")))

    if layer_raw is None or inputs_raw is None:
        return None

    layer = _as_int(layer_raw, field="layer", line_no=line_no)
    vec = _collect_input_vector(inputs_raw, field="inputs", line_no=line_no)

    if input_granularity == "layer":
        expert: int | None = None
        if expert_raw is not None:
            expert = _as_int(expert_raw, field="expert", line_no=line_no)
        return vec, layer, expert

    if expert_raw is None:
        return None
    expert = _as_int(expert_raw, field="expert", line_no=line_no)
    return vec, layer, expert


def _load_routed_trace_jsonl(
    *,
    trace_path: Path,
    args: argparse.Namespace,
) -> tuple[TraceCaptureResult, dict[str, int]]:
    if not trace_path.is_file():
        raise MolrCovarianceError(
            f"MoE routed trace JSONL not found at '{trace_path}'. "
            "Ensure llama.cpp was started with routed-input trace emission enabled.",
        )

    per_expert_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    per_layer_count: defaultdict[int, int] = defaultdict(int)
    dropped_by_cap: Counter[str] = Counter()
    inputs_rows: list[np.ndarray] = []
    layers_rows: list[int] = []
    experts_rows: list[int] = []
    d_model: int | None = None

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception as exc:
            raise MolrCovarianceError(f"Trace line {idx}: invalid JSON") from exc

        parsed = _extract_trace_fields(
            payload,
            line_no=idx,
            input_granularity=str(args.input_granularity_resolved),
        )
        if parsed is None:
            continue

        vec, layer, expert_raw = parsed
        expert = int(expert_raw) if expert_raw is not None else -1
        if d_model is None:
            d_model = int(vec.shape[0])
        elif int(vec.shape[0]) != d_model:
            raise MolrCovarianceError(
                f"Trace line {idx}: d_model mismatch; expected {d_model}, got {int(vec.shape[0])}",
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


def _build_llama_cli_command(
    *,
    args: argparse.Namespace,
    record: PromptInferenceRecord,
    record_index: int,
) -> list[str]:
    cli_bin = args.capture_llama_cli or os.environ.get("LLAMA_MOLR_CLI_BIN") or "llama-cli"

    merged_params = dict(record.inference_params)
    if "seed" not in merged_params and int(args.capture_seed) != 0:
        merged_params["seed"] = int(args.capture_seed) + int(record_index)
    if "n_predict" not in merged_params:
        merged_params["n_predict"] = int(args.tokens) if int(args.tokens) > 0 else 0

    cmd = [str(cli_bin), "-m", str(args.model), "-p", record.prompt]
    cmd.extend(["-n", str(int(merged_params.get("n_predict", 0)))])

    supported_float = {
        "temperature": "--temp",
        "top_p": "--top-p",
        "min_p": "--min-p",
        "repeat_penalty": "--repeat-penalty",
    }
    supported_int = {
        "seed": "--seed",
        "top_k": "--top-k",
        "repeat_last_n": "--repeat-last-n",
        "n_ctx": "-c",
        "n_batch": "-b",
        "n_ubatch": "-ub",
    }
    supported_bool_flag = {
        "no_display_prompt": "--no-display-prompt",
    }

    for key, flag in supported_float.items():
        if key in merged_params:
            cmd.extend([flag, str(_as_number(merged_params[key], field=key, line_no=record_index + 1))])
    for key, flag in supported_int.items():
        if key in merged_params:
            cmd.extend([flag, str(_as_int(merged_params[key], field=key, line_no=record_index + 1))])
    for key, flag in supported_bool_flag.items():
        if bool(merged_params.get(key, False)):
            cmd.append(flag)

    # Allow explicit passthrough for advanced flags not yet modeled above.
    extra_cli_args = merged_params.get("extra_cli_args", [])
    if extra_cli_args:
        if not isinstance(extra_cli_args, list) or not all(isinstance(it, str) for it in extra_cli_args):
            raise MolrCovarianceError(
                f"Record {record_index + 1}: 'extra_cli_args' must be a list of strings",
            )
        cmd.extend(extra_cli_args)

    # Keep runs quiet and deterministic for capture pipeline.
    if "--no-display-prompt" not in cmd:
        cmd.append("--no-display-prompt")
    return cmd


def _run_inference_capture_bridge(
    *,
    args: argparse.Namespace,
    prompt_spec: PromptInferenceSpec,
    trace_jsonl_path: Path,
) -> None:
    if trace_jsonl_path.exists():
        trace_jsonl_path.unlink()
    trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional prompt manifest used only for reproducibility/debug context.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl", delete=False) as tmp_file:
        temp_prompt_manifest = Path(tmp_file.name)
        for record in prompt_spec.records:
            tmp_file.write(
                json.dumps(
                    {
                        "prompt": record.prompt,
                        "inference_params": record.inference_params,
                    },
                    sort_keys=True,
                )
                + "\n",
            )

    try:
        for idx, record in enumerate(prompt_spec.records):
            cmd = _build_llama_cli_command(args=args, record=record, record_index=idx)
            env = os.environ.copy()
            env["LLAMA_MOE_TRACE_ENABLE"] = "1"
            env["LLAMA_MOE_TRACE_JSONL"] = str(trace_jsonl_path)
            env["LLAMA_MOE_TRACE_FORMAT"] = "jsonl"
            env["LLAMA_MOE_TRACE_GRANULARITY"] = str(args.input_granularity_resolved)
            env["LLAMA_MOLR_CAPTURE_SOURCE"] = str(temp_prompt_manifest)

            proc = subprocess.run(
                cmd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip()
                raise MolrCovarianceError(
                    f"capture inference failed for record {idx + 1} (exit={proc.returncode}): {stderr_tail}",
                )
    finally:
        try:
            temp_prompt_manifest.unlink(missing_ok=True)
        except Exception:
            pass


def _load_routed_contract(
    path: Path,
    *,
    expected_model: str | None,
    input_granularity: str,
) -> TraceCaptureResult:
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise MolrCovarianceError(f"Failed loading routed-inputs NPZ '{path}': {exc}") from exc

    required = ("inputs", "layers")
    if input_granularity == "expert":
        required = ("inputs", "layers", "experts")
    missing = [name for name in required if name not in payload]
    if missing:
        raise MolrCovarianceError(
            f"Routed-input NPZ '{path}' missing required arrays: {', '.join(missing)}",
        )

    inputs = np.asarray(payload["inputs"])
    layers = np.asarray(payload["layers"])
    experts = np.asarray(payload["experts"]) if "experts" in payload else None

    if inputs.ndim != 2:
        raise MolrCovarianceError(f"inputs must be 2D [N,D], got shape={inputs.shape}")
    if layers.ndim != 1:
        raise MolrCovarianceError(
            f"layers must be 1D; got layers={layers.shape}",
        )
    n_rows = inputs.shape[0]
    if layers.shape[0] != n_rows:
        raise MolrCovarianceError(
            "Row count mismatch in routed-input arrays: "
            f"inputs={n_rows}, layers={layers.shape[0]}",
        )
    if experts is not None:
        if experts.ndim != 1:
            raise MolrCovarianceError(f"experts must be 1D; got experts={experts.shape}")
        if experts.shape[0] != n_rows:
            raise MolrCovarianceError(
                "Row count mismatch in routed-input arrays: "
                f"inputs={n_rows}, layers={layers.shape[0]}, experts={experts.shape[0]}",
            )
    elif input_granularity == "layer":
        experts = np.full((n_rows,), -1, dtype=np.int64)
    else:
        raise MolrCovarianceError("expert granularity requires 'experts' array in routed-input NPZ")
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
        routed_experts=np.asarray(experts).astype(np.int64, copy=False),
    )


def _capture_routed_traces(args: argparse.Namespace) -> tuple[TraceCaptureResult, dict[str, int], PromptInferenceSpec, str]:
    prompt_spec = _load_capture_prompt_inference_spec(args)

    trace_jsonl_str = args.capture_trace_jsonl or os.environ.get("LLAMA_MOE_TRACE_JSONL", "")
    if not trace_jsonl_str:
        raise MolrCovarianceError(
            "Capture mode requires routed trace sink path via --capture-trace-jsonl or LLAMA_MOE_TRACE_JSONL.",
        )
    trace_jsonl_path = Path(trace_jsonl_str).expanduser().resolve()

    _run_inference_capture_bridge(
        args=args,
        prompt_spec=prompt_spec,
        trace_jsonl_path=trace_jsonl_path,
    )
    trace_result, dropped = _load_routed_trace_jsonl(trace_path=trace_jsonl_path, args=args)
    return trace_result, dropped, prompt_spec, str(trace_jsonl_path)


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
    granularity = str(args.input_granularity_resolved)
    schema_version = (
        MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2
        if granularity == "layer"
        else MOLR_COVARIANCE_NPZ_SCHEMA_VERSION
    )
    summary_schema = (
        MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION_V2
        if granularity == "layer"
        else MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION
    )

    npz_arrays: dict[str, Any] = {
        "schema_version": np.array(schema_version),
        "model_spec": np.array(args.model),
        "d_model": np.array(0, dtype=np.int64),
        "layers": np.zeros((0,), dtype=np.int64),
        "sample_count": np.zeros((0,), dtype=np.int64),
        "jitter_used": np.zeros((0,), dtype=np.float64),
        "mu": np.zeros((0, 0), dtype=np.float32),
        "chol": np.zeros((0, 0, 0), dtype=np.float32),
    }
    if granularity == "layer":
        npz_arrays["granularity"] = np.array("layer")
    else:
        npz_arrays["experts"] = np.zeros((0,), dtype=np.int64)

    _save_npz(
        out_npz_path,
        **npz_arrays,
    )

    summary = {
        "schema_version": summary_schema,
        "created_at_utc": _now_utc_iso(),
        "model_spec": args.model,
        "tokens_requested": int(args.tokens),
        "input_granularity": granularity,
        "input_contract": {
            "mode": mode,
            "routed_inputs_npz": str(args.routed_inputs_npz) if args.routed_inputs_npz else None,
            "routed_inputs_npz_schema": "molr_routed_inputs.v1",
        },
        "capture_runtime": {
            "status": mode,
        },
        "deprecation_warnings": list(getattr(args, "deprecation_warnings", [])),
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


def _compute_covariance_for_layer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
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


def _maybe_emit_layer_traces_npz(
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
        schema_version=np.array(MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION),
        model_spec=np.array(trace.model_spec),
        d_model=np.array(int(trace.d_model), dtype=np.int64),
        inputs=trace.routed_inputs.astype(trace_dtype, copy=False),
        layers=trace.routed_layers.astype(np.int64, copy=False),
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
        out_layer_traces_path = (
            Path(args.out_layer_traces_npz).expanduser().resolve()
            if args.out_layer_traces_npz
            else None
        )

        capture_mode = bool(args.capture_layer_traces)
        capture_runtime_status = "capture_enabled" if capture_mode else "contract_only"
        capture_dropped_counters: dict[str, int] = {}
        capture_prompt_spec: PromptInferenceSpec | None = None
        capture_trace_jsonl_used: str | None = None

        if capture_mode:
            try:
                trace_result, capture_dropped_counters, capture_prompt_spec, capture_trace_jsonl_used = _capture_routed_traces(args)
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
                        "No --routed-inputs-npz provided. Enable --capture-layer-traces with "
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
            trace_result = _load_routed_contract(
                routed_path,
                expected_model=str(args.model),
                input_granularity=str(args.input_granularity_resolved),
            )

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

        if str(args.input_granularity_resolved) == "layer":
            unique_keys_all = sorted(expert_pool.keys())
            unique_layers_all = sorted(layer_pool.keys())
            unique_layers = unique_layers_all if args.max_experts <= 0 else unique_layers_all[: int(args.max_experts)]

            for layer in unique_layers:
                layer_x = layer_pool.get(layer, np.zeros((0, d_model), dtype=np.float64))
                layer_count = int(layer_x.shape[0])
                if layer_count < int(args.min_samples_per_layer_effective):
                    reason = f"insufficient_layer_samples(<{int(args.min_samples_per_layer_effective)})"
                    experts_failed.append(
                        {
                            "layer": layer,
                            "sample_count": layer_count,
                            "reason": reason,
                        }
                    )
                    failure_reasons[reason] += 1
                    continue

                try:
                    mu, chol, jitter_used, cov_trace = _compute_covariance_for_layer(layer_x)
                except MolrCovarianceError as exc:
                    reason = str(exc)
                    experts_failed.append(
                        {
                            "layer": layer,
                            "sample_count": layer_count,
                            "reason": reason,
                        }
                    )
                    failure_reasons[reason] += 1
                    continue

                success_layers.append(layer)
                success_counts.append(layer_count)
                success_jitter.append(float(jitter_used))
                success_mu.append(mu.astype(np.float32, copy=False))
                success_chol.append(chol.astype(np.float32, copy=False))
                experts_succeeded.append(
                    {
                        "layer": layer,
                        "sample_count": layer_count,
                        "sample_source": "layer",
                        "effective_sample_count": layer_count,
                        "d_model": d_model,
                        "jitter_used": float(jitter_used),
                        "cov_trace": float(cov_trace),
                    }
                )

            unique_keys: list[tuple[int, int]] = []
        else:
            unique_keys_all = sorted(expert_pool.keys())
            unique_keys = unique_keys_all if args.max_experts <= 0 else unique_keys_all[: int(args.max_experts)]

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

        covariance_granularity = str(args.input_granularity_resolved)
        npz_arrays: dict[str, Any] = {
            "schema_version": np.array(
                MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2 if covariance_granularity == "layer" else MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
            ),
            "model_spec": np.array(trace_result.model_spec or str(args.model)),
            "d_model": np.array(d_model, dtype=np.int64),
            "layers": np.asarray(success_layers, dtype=np.int64),
            "sample_count": np.asarray(success_counts, dtype=np.int64),
            "jitter_used": np.asarray(success_jitter, dtype=np.float64),
            "mu": mu_arr,
            "chol": chol_arr,
        }
        if covariance_granularity == "layer":
            npz_arrays["granularity"] = np.array("layer")
        else:
            npz_arrays["experts"] = np.asarray(success_experts, dtype=np.int64)

        _save_npz(out_npz_path, **npz_arrays)

        routed_traces_path_str = (
            _maybe_emit_routed_traces_npz(
                args=args,
                trace=trace_result,
                out_path=out_routed_traces_path,
            )
            if covariance_granularity == "expert"
            else None
        )
        layer_traces_path_str = _maybe_emit_layer_traces_npz(
            args=args,
            trace=trace_result,
            out_path=out_layer_traces_path,
        )

        effective_layer_threshold = (
            int(args.min_layer_samples_for_fallback)
            if int(args.min_layer_samples_for_fallback) > 0
            else int(args.min_samples_per_expert)
        )

        input_contract: dict[str, Any] = {
            "mode": capture_runtime_status,
            "input_granularity": covariance_granularity,
            "routed_inputs_npz_schema": "molr_routed_inputs.v1",
        }
        if capture_mode:
            input_contract["capture_prompts_jsonl"] = str(Path(args.capture_prompts_jsonl).expanduser().resolve())
            input_contract["capture_prompt_inference_schema"] = {
                "formats": ["json_object_records", "jsonl_records"],
                "required_fields": ["prompt"],
                "optional_fields": ["inference_params"],
            }
            input_contract["capture_trace_schema"] = {
                "required_fields": (
                    ["layer", "expert", "inputs"]
                    if covariance_granularity == "expert"
                    else ["layer", "inputs"]
                ),
                "accepted_event_names": ["moe_layer_input", "moe_routed_input", "routed_input", "moe.trace.routed_input"],
            }
        else:
            input_contract["routed_inputs_npz"] = str(Path(args.routed_inputs_npz).expanduser().resolve())
            input_contract["required_arrays"] = (
                ["inputs", "layers", "experts"]
                if covariance_granularity == "expert"
                else ["inputs", "layers"]
            )

        summary = {
            "schema_version": (
                MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION_V2
                if covariance_granularity == "layer"
                else MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION
            ),
            "created_at_utc": _now_utc_iso(),
            "model_spec": trace_result.model_spec or str(args.model),
            "tokens_requested": int(args.tokens),
            "input_granularity": covariance_granularity,
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
                "trace_jsonl_path": capture_trace_jsonl_used,
                "prompt_records_total": (
                    int(len(capture_prompt_spec.records))
                    if (capture_mode and capture_prompt_spec is not None)
                    else None
                ),
                "prompt_inference_source_schema": (
                    capture_prompt_spec.source_schema
                    if (capture_mode and capture_prompt_spec is not None)
                    else None
                ),
            },
            "deprecation_warnings": list(getattr(args, "deprecation_warnings", [])),
            "config": {
                "min_samples_per_expert": int(args.min_samples_per_expert),
                "min_samples_per_layer": int(args.min_samples_per_layer_effective),
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
                "experts_processed_total": len(unique_keys) if covariance_granularity == "expert" else None,
                "layers_processed_total": len(success_layers) if covariance_granularity == "layer" else None,
            },
            "layers_succeeded_total": len(experts_succeeded) if covariance_granularity == "layer" else None,
            "layers_failed_total": len(experts_failed) if covariance_granularity == "layer" else None,
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
                "covariance_npz_schema_version": (
                    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2
                    if covariance_granularity == "layer"
                    else MOLR_COVARIANCE_NPZ_SCHEMA_VERSION
                ),
                "routed_traces_npz": routed_traces_path_str,
                "routed_traces_npz_schema_version": (
                    MOLR_ROUTED_TRACES_NPZ_SCHEMA_VERSION if routed_traces_path_str is not None else None
                ),
                "layer_traces_npz": layer_traces_path_str,
                "layer_traces_npz_schema_version": (
                    MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION if layer_traces_path_str is not None else None
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
