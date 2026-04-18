#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
)


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrCovarianceError(RuntimeError):
    """Raised when covariance capture contract execution fails."""


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
            "Phase-1 covariance capture scaffold. Computes per-expert (mu, Cholesky(cov)) "
            "from pre-captured routed input vectors and emits covariance_stats.npz + summary JSON."
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
        "--min-samples-per-expert",
        type=int,
        default=16,
        help="Minimum routed samples required to compute covariance per expert. Default: 16.",
    )
    parser.add_argument(
        "--max-experts",
        type=int,
        default=0,
        help="Optional cap on number of experts processed after sorting. 0 means no cap.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Allow writing empty scaffold artifacts when no --routed-inputs-npz is provided. "
            "Useful before runtime-routed capture integration exists."
        ),
    )
    parser.add_argument("--out-npz", required=True, help="Output path for covariance_stats.npz.")
    parser.add_argument("--out-json", required=True, help="Output path for covariance_summary.json.")

    args = parser.parse_args(argv)
    if args.tokens < 0:
        parser.error("--tokens must be >= 0.")
    if args.min_samples_per_expert <= 1:
        parser.error("--min-samples-per-expert must be > 1.")
    if args.max_experts < 0:
        parser.error("--max-experts must be >= 0.")
    return args


def _load_routed_contract(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if n_rows == 0:
        raise MolrCovarianceError("Routed-input contract is empty (N=0).")
    if not np.isfinite(inputs).all():
        raise MolrCovarianceError("inputs contain non-finite values.")

    return inputs.astype(np.float64, copy=False), layers.astype(np.int64, copy=False), experts.astype(np.int64, copy=False)


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


def _build_empty_artifacts(*, args: argparse.Namespace, out_npz_path: Path, out_json_path: Path, reason: str) -> None:
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
            "routed_inputs_npz": None,
            "routed_inputs_npz_schema": "molr_routed_inputs.v1",
        },
        "status": "empty",
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


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        out_npz_path = Path(args.out_npz).expanduser().resolve()
        out_json_path = Path(args.out_json).expanduser().resolve()

        if not args.routed_inputs_npz:
            if not args.allow_empty:
                raise MolrCovarianceError(
                    "No --routed-inputs-npz provided. Runtime routed-input capture integration is "
                    "outside Phase-1 scaffold scope; provide pre-captured routed inputs or pass --allow-empty.",
                )
            _build_empty_artifacts(
                args=args,
                out_npz_path=out_npz_path,
                out_json_path=out_json_path,
                reason="missing_routed_inputs_npz",
            )
            print(f"[molr-cov] wrote empty scaffold artifacts -> {out_npz_path} and {out_json_path}")
            return EXIT_OK

        routed_path = Path(args.routed_inputs_npz).expanduser().resolve()
        if not routed_path.is_file():
            raise MolrCovarianceError(f"--routed-inputs-npz path is not a file: '{routed_path}'")

        inputs, layers, experts = _load_routed_contract(routed_path)
        d_model = int(inputs.shape[1])

        unique_keys_all = sorted({(int(l), int(e)) for l, e in zip(layers.tolist(), experts.tolist())})
        unique_keys = unique_keys_all
        if args.max_experts > 0:
            unique_keys = unique_keys[: args.max_experts]

        success_layers: list[int] = []
        success_experts: list[int] = []
        success_counts: list[int] = []
        success_jitter: list[float] = []
        success_mu: list[np.ndarray] = []
        success_chol: list[np.ndarray] = []

        experts_succeeded: list[dict[str, Any]] = []
        experts_failed: list[dict[str, Any]] = []
        failure_reasons: Counter[str] = Counter()

        for layer, expert in unique_keys:
            mask = (layers == layer) & (experts == expert)
            x = inputs[mask]
            sample_count = int(x.shape[0])

            if sample_count < int(args.min_samples_per_expert):
                reason = f"insufficient_samples(<{args.min_samples_per_expert})"
                experts_failed.append({"layer": layer, "expert": expert, "sample_count": sample_count, "reason": reason})
                failure_reasons[reason] += 1
                continue

            mu = np.mean(x, axis=0, dtype=np.float64)
            centered = x - mu
            cov = (centered.T @ centered) / float(sample_count - 1)

            try:
                chol, jitter_used = _cholesky_with_jitter(cov)
            except MolrCovarianceError as exc:
                reason = str(exc)
                experts_failed.append({"layer": layer, "expert": expert, "sample_count": sample_count, "reason": reason})
                failure_reasons[reason] += 1
                continue

            success_layers.append(layer)
            success_experts.append(expert)
            success_counts.append(sample_count)
            success_jitter.append(float(jitter_used))
            success_mu.append(mu.astype(np.float32, copy=False))
            success_chol.append(chol.astype(np.float32, copy=False))

            experts_succeeded.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "sample_count": sample_count,
                    "d_model": d_model,
                    "jitter_used": float(jitter_used),
                    "cov_trace": float(np.trace(cov, dtype=np.float64)),
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
            model_spec=np.array(args.model),
            d_model=np.array(d_model, dtype=np.int64),
            layers=np.asarray(success_layers, dtype=np.int64),
            experts=np.asarray(success_experts, dtype=np.int64),
            sample_count=np.asarray(success_counts, dtype=np.int64),
            jitter_used=np.asarray(success_jitter, dtype=np.float64),
            mu=mu_arr,
            chol=chol_arr,
        )

        summary = {
            "schema_version": MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": args.model,
            "tokens_requested": int(args.tokens),
            "input_contract": {
                "routed_inputs_npz": str(routed_path),
                "routed_inputs_npz_schema": "molr_routed_inputs.v1",
                "required_arrays": ["inputs", "layers", "experts"],
            },
            "capture_runtime": {
                "status": "contract_only",
                "note": (
                    "This scaffold consumes pre-captured routed inputs. "
                    "Direct model-token routing capture is a later integration step."
                ),
            },
            "config": {
                "min_samples_per_expert": int(args.min_samples_per_expert),
                "max_experts": int(args.max_experts),
                "cholesky_jitter_schedule": list(CHOLESKY_JITTER_SCHEDULE),
            },
            "observed": {
                "rows_total": int(inputs.shape[0]),
                "d_model": d_model,
                "experts_observed_total": len(unique_keys_all),
                "experts_processed_total": len(unique_keys),
            },
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
