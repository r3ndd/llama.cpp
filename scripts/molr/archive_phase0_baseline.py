#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.artifacts import (
    MolrPhase0Error,
    add_common_archive_args,
    build_coverage_summary,
    build_phase0_run_metadata,
    ensure_expected_svd_report,
    load_json,
    sha256_file,
    stage_phase0_artifacts,
    validate_common_archive_args,
)


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2
EXIT_IO_ERROR = 3


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive Phase-0 MoLR baseline artifacts from an existing svd_report.json "
            "and emit operator-focused run metadata."
        ),
    )
    add_common_archive_args(parser)
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail if coverage plausibility checks are not in pass state.",
    )
    args = parser.parse_args(argv)
    validate_common_archive_args(args, parser)
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        svd_report_path = Path(args.svd_report).expanduser().resolve()
        run_dir = Path(args.run_dir).expanduser().resolve()

        if not svd_report_path.is_file():
            raise MolrPhase0Error(f"svd_report path is not a file: '{svd_report_path}'")

        svd_report = load_json(svd_report_path)
        ensure_expected_svd_report(
            svd_report=svd_report,
            expected_model_spec=args.model,
            allow_model_mismatch=args.allow_model_mismatch,
        )
        run_model_spec = str(svd_report.get("run", {}).get("model_spec") or args.model)

        coverage = build_coverage_summary(svd_report)
        if args.strict_coverage and coverage.plausibility_status != "pass":
            reason = "; ".join(coverage.plausibility_reasons) or "unknown reason"
            raise MolrPhase0Error(
                "Coverage plausibility check failed in strict mode: "
                f"status={coverage.plausibility_status}; {reason}",
            )

        run_metadata = build_phase0_run_metadata(
            model_spec=run_model_spec,
            workers=args.workers,
            blas_threads=args.blas_threads,
            svd_report_path=svd_report_path,
            svd_report_sha256=sha256_file(svd_report_path),
            svd_report_schema_version=str(svd_report.get("schema_version") or ""),
            coverage=coverage,
            operator_notes=args.notes,
        )

        written = stage_phase0_artifacts(
            svd_report_path=svd_report_path,
            run_dir=run_dir,
            run_metadata=run_metadata,
        )

        print("[phase0] archived baseline artifacts:")
        for path in written:
            print(f"  - {path}")
        print(
            "[phase0] coverage: "
            f"status={coverage.plausibility_status} "
            f"candidates={coverage.candidates} analyzed={coverage.analyzed} "
            f"layers={coverage.unique_layer_count} experts={coverage.unique_expert_count}",
        )
        if coverage.plausibility_reasons:
            print("[phase0] coverage notes:")
            for reason in coverage.plausibility_reasons:
                print(f"  - {reason}")

        quant_caveat = run_metadata.get("quantization_caveat")
        if quant_caveat:
            print(f"[phase0] quantization caveat: {quant_caveat}")

        return EXIT_OK

    except MolrPhase0Error as exc:
        print(f"[error:phase0] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR
    except Exception as exc:
        print(f"[error:phase0-io] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
