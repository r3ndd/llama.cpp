#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.types import (
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2,
    MOLR_FAILURE_LEDGER_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
    MOLR_VALIDATION_REPORT_SCHEMA_VERSION,
)


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrTrainAllError(RuntimeError):
    """Raised when multi-expert MoLR training orchestration fails."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrTrainAllError(f"Failed reading JSON '{path}': {exc}") from exc


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrTrainAllError(f"Failed writing JSON '{path}': {exc}") from exc


def _npz_scalar_string(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key])
    if value.ndim == 0:
        return str(value.item())
    return str(value)


def _load_cov_keys(path: Path) -> tuple[set[tuple[int, int]], set[int]]:
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise MolrTrainAllError(f"Failed loading covariance NPZ '{path}': {exc}") from exc

    schema_version = _npz_scalar_string(payload, "schema_version")
    accepted_versions = {
        MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
        MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2,
    }
    if schema_version is not None and schema_version not in accepted_versions:
        raise MolrTrainAllError(
            f"Covariance NPZ schema mismatch: got '{schema_version}', expected one of {sorted(accepted_versions)}",
        )

    if "layers" not in payload:
        raise MolrTrainAllError(
            f"Covariance NPZ '{path}' missing required array: layers",
        )

    layers = np.asarray(payload["layers"]).astype(np.int64, copy=False)
    if layers.ndim != 1:
        raise MolrTrainAllError(f"Invalid covariance layer key array: layers={layers.shape}")

    granularity = _npz_scalar_string(payload, "granularity")
    layer_keys: set[int] = set()

    if "experts" not in payload:
        layer_keys = {int(layer) for layer in layers.tolist()}
        return set(), layer_keys

    experts = np.asarray(payload["experts"]).astype(np.int64, copy=False)
    if experts.ndim != 1 or layers.shape[0] != experts.shape[0]:
        raise MolrTrainAllError(f"Invalid covariance key arrays: layers={layers.shape}, experts={experts.shape}")

    if granularity == "layer":
        layer_keys = {int(layer) for layer in layers.tolist()}

    return {(int(layer), int(expert)) for layer, expert in zip(layers.tolist(), experts.tolist())}, layer_keys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate Phase-2 per-expert MoLR training across all experts in molr_plan.json "
            "and emit merged validation report + failure ledger."
        ),
    )
    parser.add_argument("--model", required=True, help="Model spec for metadata and subcommand invocation.")
    parser.add_argument("--plan-json", required=True, help="Path to molr_plan.json.")
    parser.add_argument("--cov-npz", required=True, help="Path to covariance_stats.npz.")
    parser.add_argument(
        "--weights-dir",
        required=True,
        help=(
            "Directory containing per-expert full weights NPZ files for offline training. "
            "Default lookup name: expert_weights_{layer}_{expert}.npz"
        ),
    )
    parser.add_argument(
        "--weights-pattern",
        default="expert_weights_{layer}_{expert}.npz",
        help=(
            "Filename pattern inside --weights-dir for expert full weights. "
            "Allowed placeholders: {layer}, {expert}."
        ),
    )
    parser.add_argument("--steps", type=int, default=20000, help="Training steps per expert.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size per expert.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--lambda-lb", type=float, default=0.01, help="Load-balance coefficient.")
    parser.add_argument("--lambda-err", type=float, default=0.05, help="Error-head coefficient.")
    parser.add_argument("--validation-samples", type=int, default=2048, help="Validation synthetic samples.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--cosine-threshold", type=float, default=0.95, help="Cosine validation threshold.")
    parser.add_argument(
        "--error-corr-threshold",
        type=float,
        default=0.70,
        help="Error-head Pearson r validation threshold.",
    )
    parser.add_argument(
        "--max-experts",
        type=int,
        default=0,
        help="Optional cap on experts processed from plan order. 0 means no cap.",
    )
    parser.add_argument(
        "--continue-on-train-error",
        action="store_true",
        help="Continue processing remaining experts if one training subprocess fails.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for invoking train_expert_molr.py. Default: current interpreter.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints/validation/merged reports.")

    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.lr <= 0.0:
        parser.error("--lr must be > 0")
    if args.lambda_lb < 0.0:
        parser.error("--lambda-lb must be >= 0")
    if args.lambda_err < 0.0:
        parser.error("--lambda-err must be >= 0")
    if args.validation_samples <= 0:
        parser.error("--validation-samples must be > 0")
    if args.max_experts < 0:
        parser.error("--max-experts must be >= 0")
    return args


def _weights_path(weights_dir: Path, pattern: str, *, layer: int, expert: int) -> Path:
    filename = pattern.format(layer=layer, expert=expert)
    return (weights_dir / filename).resolve()


def _run_train_expert(
    *,
    python_exe: str,
    script_path: Path,
    model: str,
    plan_json: Path,
    cov_npz: Path,
    weights_npz: Path,
    layer: int,
    expert: int,
    steps: int,
    batch_size: int,
    lr: float,
    lambda_lb: float,
    lambda_err: float,
    validation_samples: int,
    seed: int,
    cosine_threshold: float,
    error_corr_threshold: float,
    out_checkpoint: Path,
    out_validation: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        python_exe,
        str(script_path),
        "--model",
        model,
        "--plan-json",
        str(plan_json),
        "--cov-npz",
        str(cov_npz),
        "--weights-npz",
        str(weights_npz),
        "--layer",
        str(layer),
        "--expert",
        str(expert),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--lambda-lb",
        str(lambda_lb),
        "--lambda-err",
        str(lambda_err),
        "--validation-samples",
        str(validation_samples),
        "--seed",
        str(seed),
        "--cosine-threshold",
        str(cosine_threshold),
        "--error-corr-threshold",
        str(error_corr_threshold),
        "--out-checkpoint",
        str(out_checkpoint),
        "--out-validation",
        str(out_validation),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        plan_path = Path(args.plan_json).expanduser().resolve()
        cov_path = Path(args.cov_npz).expanduser().resolve()
        weights_dir = Path(args.weights_dir).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve()
        checkpoints_dir = out_dir / "checkpoints"
        validations_dir = out_dir / "validation"
        report_path = out_dir / "molr_validation_report.json"
        ledger_path = out_dir / "molr_failure_ledger.json"

        if not plan_path.is_file():
            raise MolrTrainAllError(f"--plan-json path is not a file: '{plan_path}'")
        if not cov_path.is_file():
            raise MolrTrainAllError(f"--cov-npz path is not a file: '{cov_path}'")
        if not weights_dir.is_dir():
            raise MolrTrainAllError(f"--weights-dir path is not a directory: '{weights_dir}'")

        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        validations_dir.mkdir(parents=True, exist_ok=True)

        plan = _load_json(plan_path)
        if str(plan.get("schema_version") or "") != MOLR_PLAN_SCHEMA_VERSION:
            raise MolrTrainAllError(
                f"Unexpected plan schema_version='{plan.get('schema_version')}', expected '{MOLR_PLAN_SCHEMA_VERSION}'",
            )
        plan_model_spec = str(plan.get("model_spec") or "")
        if plan_model_spec and plan_model_spec != args.model:
            raise MolrTrainAllError(
                f"Model spec mismatch: --model='{args.model}' does not match plan model_spec='{plan_model_spec}'",
            )

        experts_plan = [
            {"layer": int(entry.get("layer")), "expert": int(entry.get("expert"))}
            for entry in plan.get("experts", [])
        ]
        experts_plan.sort(key=lambda x: (x["layer"], x["expert"]))
        if args.max_experts > 0:
            experts_plan = experts_plan[: int(args.max_experts)]

        cov_keys, cov_layer_keys = _load_cov_keys(cov_path)
        script_path = Path(__file__).resolve().parent / "train_expert_molr.py"
        if not script_path.is_file():
            raise MolrTrainAllError(f"train_expert_molr.py not found at '{script_path}'")

        validation_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        aborted_early = False

        for idx, expert_ref in enumerate(experts_plan):
            layer = expert_ref["layer"]
            expert = expert_ref["expert"]
            key = (layer, expert)

            weights_npz = _weights_path(weights_dir, args.weights_pattern, layer=layer, expert=expert)
            checkpoint_out = checkpoints_dir / f"molr_expert_{layer}_{expert}.npz"
            validation_out = validations_dir / f"molr_validation_{layer}_{expert}.json"

            has_cov = (key in cov_keys) or (layer in cov_layer_keys)
            if not has_cov:
                failure_rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "status": "train_skipped_missing_cov",
                        "reason": "missing_covariance_entry",
                    }
                )
                print(
                    "[molr-train-all] skip "
                    f"layer={layer} expert={expert} reason=missing_covariance_entry",
                )
                continue

            if not weights_npz.is_file():
                failure_rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "status": "train_skipped_missing_weights",
                        "reason": "missing_weights_npz",
                        "weights_path": str(weights_npz),
                    }
                )
                print(
                    "[molr-train-all] skip "
                    f"layer={layer} expert={expert} reason=missing_weights_npz",
                )
                continue

            result = _run_train_expert(
                python_exe=args.python,
                script_path=script_path,
                model=args.model,
                plan_json=plan_path,
                cov_npz=cov_path,
                weights_npz=weights_npz,
                layer=layer,
                expert=expert,
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                lambda_lb=float(args.lambda_lb),
                lambda_err=float(args.lambda_err),
                validation_samples=int(args.validation_samples),
                seed=int(args.seed) + idx,
                cosine_threshold=float(args.cosine_threshold),
                error_corr_threshold=float(args.error_corr_threshold),
                out_checkpoint=checkpoint_out,
                out_validation=validation_out,
            )

            if result.returncode != 0:
                failure_rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "status": "train_failed_subprocess",
                        "reason": "subprocess_nonzero_exit",
                        "returncode": int(result.returncode),
                        "stderr_tail": result.stderr.strip()[-2000:],
                    }
                )
                print(
                    "[molr-train-all] fail "
                    f"layer={layer} expert={expert} returncode={result.returncode}",
                    file=sys.stderr,
                )
                if not args.continue_on_train_error:
                    aborted_early = True
                    break
                continue

            try:
                validation_payload = _load_json(validation_out)
            except MolrTrainAllError as exc:
                failure_rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "status": "train_failed_validation_read",
                        "reason": str(exc),
                    }
                )
                if not args.continue_on_train_error:
                    aborted_early = True
                    break
                continue

            metrics = validation_payload.get("validation_metrics", {})
            status = str(validation_payload.get("status") or "unknown")
            validation_rows.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "status": status,
                    "failure_reasons": validation_payload.get("failure_reasons", []),
                    "checkpoint_npz": str(checkpoint_out),
                    "validation_json": str(validation_out),
                    "metrics": {
                        "cosine_similarity_mean": float(metrics.get("cosine_similarity_mean", 0.0)),
                        "relative_output_norm_error_mean": float(metrics.get("relative_output_norm_error_mean", 0.0)),
                        "router_entropy_mean": float(metrics.get("router_entropy_mean", 0.0)),
                        "error_head_pearson_r": float(metrics.get("error_head_pearson_r", 0.0)),
                    },
                }
            )

        succeeded = [row for row in validation_rows if row["status"] == "pass"]
        failed_quality = [row for row in validation_rows if row["status"] != "pass"]
        all_failures = [*failure_rows]
        for row in failed_quality:
            all_failures.append(
                {
                    "layer": row["layer"],
                    "expert": row["expert"],
                    "status": "validation_failed",
                    "reason": "failed_quality_thresholds",
                    "failure_reasons": row.get("failure_reasons", []),
                }
            )

        def _mean_metric(name: str) -> float:
            if not validation_rows:
                return 0.0
            values = [float(row["metrics"].get(name, 0.0)) for row in validation_rows]
            return float(sum(values) / len(values))

        report_payload = {
            "schema_version": MOLR_VALIDATION_REPORT_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": args.model,
            "inputs": {
                "plan_json": str(plan_path),
                "covariance_npz": str(cov_path),
                "weights_dir": str(weights_dir),
                "weights_pattern": args.weights_pattern,
            },
            "config": {
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "lambda_lb": float(args.lambda_lb),
                "lambda_err": float(args.lambda_err),
                "validation_samples": int(args.validation_samples),
                "seed": int(args.seed),
                "cosine_threshold": float(args.cosine_threshold),
                "error_corr_threshold": float(args.error_corr_threshold),
                "max_experts": int(args.max_experts),
            },
            "summary": {
                "experts_in_plan_total": len(plan.get("experts", [])),
                "experts_attempted_total": len(validation_rows) + len(failure_rows),
                "experts_trained_total": len(validation_rows),
                "experts_pass_total": len(succeeded),
                "experts_validation_fail_total": len(failed_quality),
                "experts_orchestration_fail_total": len(failure_rows),
                "aborted_early": bool(aborted_early),
                "metrics_mean": {
                    "cosine_similarity_mean": _mean_metric("cosine_similarity_mean"),
                    "relative_output_norm_error_mean": _mean_metric("relative_output_norm_error_mean"),
                    "router_entropy_mean": _mean_metric("router_entropy_mean"),
                    "error_head_pearson_r": _mean_metric("error_head_pearson_r"),
                },
            },
            "experts": validation_rows,
            "outputs": {
                "validation_report_json": str(report_path),
                "failure_ledger_json": str(ledger_path),
                "checkpoints_dir": str(checkpoints_dir),
                "validation_dir": str(validations_dir),
            },
        }

        ledger_payload = {
            "schema_version": MOLR_FAILURE_LEDGER_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": args.model,
            "summary": {
                "entries_total": len(all_failures),
                "orchestration_failures_total": len(failure_rows),
                "validation_failures_total": len(failed_quality),
            },
            "entries": all_failures,
            "inputs": {
                "plan_json": str(plan_path),
                "covariance_npz": str(cov_path),
                "weights_dir": str(weights_dir),
            },
        }

        _save_json(report_path, report_payload)
        _save_json(ledger_path, ledger_payload)

        print(
            "[molr-train-all] complete "
            f"trained={len(validation_rows)} pass={len(succeeded)} "
            f"validation_fail={len(failed_quality)} orchestration_fail={len(failure_rows)} "
            f"-> {report_path}",
        )

        if aborted_early:
            return EXIT_VALIDATION_ERROR
        return EXIT_OK

    except MolrTrainAllError as exc:
        print(f"[error:molr-train-all] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
