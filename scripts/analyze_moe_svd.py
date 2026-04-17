#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np

from moe_svd.gguf_discovery import (
    DiscoveryError,
    discover_expert_matrices,
    load_matrix_from_reader,
    open_gguf_reader,
)
from moe_svd.model_resolver import ModelResolutionError, resolve_model_path
from moe_svd.reporting import ReportingError, print_cli_summary, write_json_report
from moe_svd.stats import compute_summary
from moe_svd.svd_metrics import SPECTRAL_ENERGY_RANK_FRACTIONS, analyze_matrix
from moe_svd.types import FailedMatrix, PerMatrixRecord, Report, SummaryDistribution, SummaryStats


EXIT_OK = 0
EXIT_MODEL_RESOLUTION = 3
EXIT_DISCOVERY = 4
EXIT_ANALYSIS_ALL_FAILED = 5
EXIT_WRITE_ERROR = 6


def _set_blas_thread_env(threads: int) -> None:
    value = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value


def _worker_init(blas_threads: int) -> None:
    _set_blas_thread_env(blas_threads)


def _build_record(
    *,
    ref,
    metrics,
    elapsed: float,
) -> PerMatrixRecord:
    return PerMatrixRecord(
        tensor=ref.tensor_name,
        source_tensor=ref.source_tensor_name,
        layer=ref.layer,
        expert=ref.expert,
        role=ref.role,
        tensor_type=ref.tensor_type,
        packed_expert_index=ref.packed_expert_index,
        packed_expert_axis=ref.packed_expert_axis,
        shape=ref.shape,
        rank_used=metrics.rank_used,
        singular_value_count=metrics.singular_value_count,
        participation_ratio=metrics.participation_ratio,
        explained_spectral_energy_rank_fractions=metrics.explained_spectral_energy_rank_fractions,
        fro_norm=metrics.fro_norm,
        elapsed_seconds=elapsed,
        warnings=metrics.analysis_warnings,
    )


def _build_failed(ref, reason: str) -> FailedMatrix:
    return FailedMatrix(
        tensor=ref.tensor_name,
        source_tensor=ref.source_tensor_name,
        layer=ref.layer,
        expert=ref.expert,
        role=ref.role,
        packed_expert_index=ref.packed_expert_index,
        packed_expert_axis=ref.packed_expert_axis,
        reason=reason,
    )


def _partition_candidates_by_source(candidates, workers: int) -> list[list]:
    grouped: dict[str, list] = {}
    for ref in candidates:
        grouped.setdefault(ref.source_tensor_name, []).append(ref)

    groups = list(grouped.values())
    groups.sort(
        key=lambda batch: sum(r.shape[0] * r.shape[1] for r in batch),
        reverse=True,
    )

    bucket_refs: list[list] = [[] for _ in range(workers)]
    bucket_loads = [0 for _ in range(workers)]

    for batch in groups:
        idx = min(range(workers), key=lambda i: bucket_loads[i])
        bucket_refs[idx].extend(batch)
        bucket_loads[idx] += sum(r.shape[0] * r.shape[1] for r in batch)

    return [bucket for bucket in bucket_refs if bucket]


def _analyze_batch(
    *,
    gguf_path: str,
    dtype: str,
    rank_frac: float,
    refs: list,
) -> tuple[list[PerMatrixRecord], list[FailedMatrix]]:
    local_per_matrix: list[PerMatrixRecord] = []
    local_failed: list[FailedMatrix] = []
    load_cache: dict[str, object] = {}

    reader = open_gguf_reader(gguf_path)

    for ref in refs:
        t0 = time.perf_counter()
        try:
            matrix = load_matrix_from_reader(
                reader=reader,
                matrix_ref=ref,
                dtype=dtype,
                cache=load_cache,
            )
            if not np.isfinite(matrix).all():
                raise ValueError("non_finite_values_after_dequantization")

            metrics = analyze_matrix(matrix=matrix, rank_frac=rank_frac)
            elapsed = time.perf_counter() - t0
            local_per_matrix.append(_build_record(ref=ref, metrics=metrics, elapsed=elapsed))
        except Exception as exc:
            local_failed.append(_build_failed(ref, f"{type(exc).__name__}: {exc}"))

    return local_per_matrix, local_failed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SVD compressibility of MoE expert matrices in GGUF models.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model specification in <repo_id>:<filename_or_quant> format.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache root override.",
    )
    parser.add_argument(
        "--rank-frac",
        type=float,
        default=0.05,
        help="Low-rank reconstruction fraction in (0, 1]. Default: 0.05.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Dequantized compute dtype.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="Regex include filter for tensor names. Repeatable.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Regex exclude filter for tensor names. Repeatable.",
    )
    parser.add_argument(
        "--max-matrices",
        type=int,
        default=None,
        help="Optional cap for number of matrices to analyze.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first per-matrix failure.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce terminal output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for matrix analysis. Default: 1.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads per process. Default: 1.",
    )
    parser.add_argument(
        "--full-svd",
        action="store_true",
        help="Explicitly select full SVD mode (required by pilot design).",
    )

    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.rank_frac <= 0.0 or args.rank_frac > 1.0:
        parser.error("--rank-frac must be in (0, 1].")

    if args.max_matrices is not None and args.max_matrices <= 0:
        parser.error("--max-matrices must be > 0 when provided.")

    if args.workers <= 0:
        parser.error("--workers must be > 0.")

    if args.blas_threads <= 0:
        parser.error("--blas-threads must be > 0.")

    if not args.full_svd:
        parser.error("--full-svd is required for this fidelity-first analysis utility.")


def _empty_summary(total_tensors: int, candidates: int, skipped_reasons: Counter[str]) -> SummaryStats:
    empty = SummaryDistribution(
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
    return SummaryStats(
        participation_ratio=empty,
        spectral_energy_rank_fractions=list(SPECTRAL_ENERGY_RANK_FRACTIONS),
        explained_spectral_energy_rank_fractions_mean=[0.0 for _ in SPECTRAL_ENERGY_RANK_FRACTIONS],
        counts={
            "total_tensors": total_tensors,
            "candidates": candidates,
            "analyzed": 0,
            "skipped_by_reason": dict(sorted(skipped_reasons.items())),
            "failed_by_reason": {},
        },
    )


def _build_report(
    *,
    args: argparse.Namespace,
    resolved,
    discovery,
    per_matrix: list[PerMatrixRecord],
    failed: list[FailedMatrix],
    total_runtime_s: float,
    summary: SummaryStats,
) -> Report:
    now = datetime.now(timezone.utc).isoformat()
    run = {
        "timestamp": now,
        "model_spec": args.model,
        "repo_id": resolved.repo_id,
        "filename": resolved.filename,
        "resolved_path": resolved.local_path,
        "downloaded": resolved.downloaded,
        "cache_dir_used": resolved.cache_dir_used,
        "rank_fraction": args.rank_frac,
        "dtype": args.dtype,
        "fidelity_mode": "full_svd",
        "workers": args.workers,
        "blas_threads": args.blas_threads,
        "max_matrices": args.max_matrices,
        "fail_fast": args.fail_fast,
        "runtime_seconds": total_runtime_s,
        "hostname": os.uname().nodename,
        "git_commit": _get_git_commit(),
    }

    discovery_payload = {
        "total_tensors": discovery.total_tensors,
        "expert_candidate_tensors": len(discovery.candidates),
        "analyzed_matrices": len(per_matrix),
        "skipped": [asdict(s) for s in discovery.skipped],
        "metadata": discovery.metadata,
    }

    caveats = [
        "Pilot run may use quantized GGUF weights (e.g. Q4_K_M). Quantization can bias singular spectra and compressibility metrics relative to FP16/BF16 checkpoints.",
        "Analysis uses exact full-spectrum singular values with compute_uv=False; this is mathematically equivalent for current metrics while reducing overhead.",
        "Parallel workers use process-based execution and should be paired with conservative BLAS thread settings to avoid oversubscription.",
    ]

    return Report(
        schema_version="1.1",
        run=run,
        discovery=discovery_payload,
        per_matrix=per_matrix,
        failed_matrices=failed,
        summary=summary,
        assumptions_and_caveats=caveats,
    )


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    started = time.perf_counter()

    try:
        resolved = resolve_model_path(args.model, args.cache_dir)
    except ModelResolutionError as exc:
        print(f"[error:model-resolution] {exc}", file=sys.stderr)
        return EXIT_MODEL_RESOLUTION

    try:
        discovery = discover_expert_matrices(
            gguf_path=resolved.local_path,
            include=args.include_pattern,
            exclude=args.exclude_pattern,
        )
    except DiscoveryError as exc:
        print(f"[error:discovery] {exc}", file=sys.stderr)
        return EXIT_DISCOVERY

    if not discovery.candidates:
        print("[error:discovery] No expert matrices discovered after filtering.", file=sys.stderr)
        return EXIT_DISCOVERY

    if args.max_matrices is not None:
        candidates = discovery.candidates[: args.max_matrices]
    else:
        candidates = discovery.candidates

    per_matrix: list[PerMatrixRecord] = []
    failed: list[FailedMatrix] = []
    skipped_reasons = Counter(s.reason for s in discovery.skipped)

    _set_blas_thread_env(args.blas_threads)

    if not args.quiet:
        print(
            "Discovery summary: "
            f"total_tensors={discovery.total_tensors} "
            f"candidates={len(candidates)} "
            f"skipped={len(discovery.skipped)}",
        )

    if args.workers == 1 or args.fail_fast:
        load_cache: dict[str, object] = {}
        try:
            reader = open_gguf_reader(resolved.local_path)
        except DiscoveryError as exc:
            print(f"[error:discovery] {exc}", file=sys.stderr)
            return EXIT_DISCOVERY

        for idx, ref in enumerate(candidates, start=1):
            t0 = time.perf_counter()
            try:
                matrix = load_matrix_from_reader(
                    reader=reader,
                    matrix_ref=ref,
                    dtype=args.dtype,
                    cache=load_cache,
                )
                if not np.isfinite(matrix).all():
                    raise ValueError("non_finite_values_after_dequantization")

                metrics = analyze_matrix(matrix=matrix, rank_frac=args.rank_frac)
                elapsed = time.perf_counter() - t0
                per_matrix.append(_build_record(ref=ref, metrics=metrics, elapsed=elapsed))
            except Exception as exc:
                failed.append(_build_failed(ref, f"{type(exc).__name__}: {exc}"))
                if args.fail_fast:
                    break
            finally:
                if not args.quiet and idx % 10 == 0:
                    print(f"Processed {idx}/{len(candidates)} matrices...")
    else:
        batches = _partition_candidates_by_source(candidates, min(args.workers, len(candidates)))
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(batches)),
            initializer=_worker_init,
            initargs=(args.blas_threads,),
            mp_context=mp_ctx,
        ) as executor:
            future_map = {
                executor.submit(
                    _analyze_batch,
                    gguf_path=resolved.local_path,
                    dtype=args.dtype,
                    rank_frac=args.rank_frac,
                    refs=batch,
                ): batch
                for batch in batches
            }

            processed = 0
            for future in as_completed(future_map):
                batch = future_map[future]
                try:
                    batch_per_matrix, batch_failed = future.result()
                except Exception as exc:
                    batch_per_matrix = []
                    batch_failed = [_build_failed(ref, f"WorkerError: {exc}") for ref in batch]

                per_matrix.extend(batch_per_matrix)
                failed.extend(batch_failed)
                processed += len(batch)
                if not args.quiet:
                    print(f"Processed {processed}/{len(candidates)} matrices...")

    per_matrix.sort(key=lambda r: (r.layer if r.layer is not None else -1, r.expert if r.expert is not None else -1, r.tensor))
    failed.sort(key=lambda r: (r.layer if r.layer is not None else -1, r.expert if r.expert is not None else -1, r.tensor))

    total_runtime_s = time.perf_counter() - started

    if per_matrix:
        summary = compute_summary(
            per_matrix,
            total_tensors=discovery.total_tensors,
            candidates=len(candidates),
            skipped_reasons=skipped_reasons,
            failed=failed,
        )
    else:
        summary = _empty_summary(
            total_tensors=discovery.total_tensors,
            candidates=len(candidates),
            skipped_reasons=skipped_reasons,
        )

    report = _build_report(
        args=args,
        resolved=resolved,
        discovery=discovery,
        per_matrix=per_matrix,
        failed=failed,
        total_runtime_s=total_runtime_s,
        summary=summary,
    )

    try:
        write_json_report(report, args.out_json)
    except ReportingError as exc:
        print(f"[error:write] {exc}", file=sys.stderr)
        return EXIT_WRITE_ERROR

    print_cli_summary(report, quiet=args.quiet)

    if not per_matrix:
        print("[error:analysis] All candidate matrices failed analysis.", file=sys.stderr)
        return EXIT_ANALYSIS_ALL_FAILED

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
