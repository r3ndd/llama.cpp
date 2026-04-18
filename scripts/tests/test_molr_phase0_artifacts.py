from __future__ import annotations

import json
from pathlib import Path

import pytest

from molr.artifacts import (
    MolrPhase0Error,
    build_coverage_summary,
    build_phase0_run_metadata,
    ensure_expected_svd_report,
    sha256_file,
    stage_phase0_artifacts,
)
from molr.archive_phase0_baseline import main as archive_phase0_main


def _base_svd_report() -> dict:
    return {
        "schema_version": "1.1",
        "run": {
            "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "fidelity_mode": "full_svd",
            "workers": 16,
            "blas_threads": 1,
        },
        "discovery": {
            "total_tensors": 100,
            "expert_candidate_tensors": 12,
        },
        "summary": {
            "counts": {
                "total_tensors": 100,
                "candidates": 12,
                "analyzed": 12,
            }
        },
        "failed_matrices": [],
        "per_matrix": [
            {"layer": 0, "expert": 0, "role": "gate"},
            {"layer": 0, "expert": 1, "role": "up"},
            {"layer": 1, "expert": 0, "role": "down"},
        ],
        "assumptions_and_caveats": [
            "Pilot run may use quantized GGUF weights (e.g. Q4_K_M).",
        ],
    }


def test_build_coverage_summary_passes_on_nonzero_plausible_coverage() -> None:
    coverage = build_coverage_summary(_base_svd_report())

    assert coverage.plausibility_status == "pass"
    assert coverage.candidates == 12
    assert coverage.analyzed == 12
    assert coverage.unique_layer_count == 2
    assert coverage.unique_expert_count == 2
    assert coverage.role_histogram["gate"] == 1


def test_build_coverage_summary_warns_when_analysis_ratio_is_low() -> None:
    report = _base_svd_report()
    report["summary"]["counts"]["analyzed"] = 3

    coverage = build_coverage_summary(report)

    assert coverage.plausibility_status == "warn"
    assert any("Less than 50%" in reason for reason in coverage.plausibility_reasons)


def test_ensure_expected_svd_report_rejects_schema_or_model_mismatch() -> None:
    report = _base_svd_report()

    with pytest.raises(MolrPhase0Error, match="schema_version"):
        ensure_expected_svd_report(
            svd_report={**report, "schema_version": "broken"},
            expected_model_spec=report["run"]["model_spec"],
            allow_model_mismatch=False,
        )

    with pytest.raises(MolrPhase0Error, match="model_spec mismatch"):
        ensure_expected_svd_report(
            svd_report=report,
            expected_model_spec="other/model:Q4_K_M",
            allow_model_mismatch=False,
        )


def test_build_phase0_run_metadata_emits_q4_caveat() -> None:
    report = _base_svd_report()
    coverage = build_coverage_summary(report)
    metadata = build_phase0_run_metadata(
        model_spec=report["run"]["model_spec"],
        workers=16,
        blas_threads=1,
        svd_report_path=Path("/tmp/svd_report.json"),
        svd_report_sha256="abc",
        svd_report_schema_version="1.1",
        coverage=coverage,
        operator_notes="",
    )

    assert metadata["schema_version"] == "phase0_run_metadata.v1"
    assert metadata["analysis_fidelity_mode"] == "full_svd"
    assert metadata["quantization"] == "Q4_K_M"
    assert "Q4_K_M quantization may bias singular spectra" in metadata["quantization_caveat"]


def test_stage_phase0_artifacts_writes_archived_files(tmp_path) -> None:
    report = _base_svd_report()
    source_svd = tmp_path / "source_svd_report.json"
    source_svd.write_text(json.dumps(report), encoding="utf-8")

    run_metadata = {
        "schema_version": "phase0_run_metadata.v1",
        "model_spec": report["run"]["model_spec"],
    }

    run_dir = tmp_path / "phase0-run"
    written = stage_phase0_artifacts(
        svd_report_path=source_svd,
        run_dir=run_dir,
        run_metadata=run_metadata,
    )

    assert len(written) == 3
    assert (run_dir / "svd_report.json").is_file()
    assert (run_dir / "run_metadata.json").is_file()
    manifest_path = run_dir / "phase0_manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_names = {entry["name"] for entry in manifest["artifacts"]}
    assert artifact_names == {"svd_report.json", "run_metadata.json"}

    archived_svd_hash = sha256_file(run_dir / "svd_report.json")
    assert any(
        entry["name"] == "svd_report.json" and entry["sha256"] == archived_svd_hash
        for entry in manifest["artifacts"]
    )


def test_archive_cli_uses_report_model_when_mismatch_is_allowed(tmp_path) -> None:
    report = _base_svd_report()
    report["run"]["model_spec"] = "other/repo:Q8_0"
    source_svd = tmp_path / "source_svd_report.json"
    source_svd.write_text(json.dumps(report), encoding="utf-8")

    run_dir = tmp_path / "phase0-run"
    exit_code = archive_phase0_main(
        [
            "--svd-report",
            str(source_svd),
            "--run-dir",
            str(run_dir),
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--allow-model-mismatch",
        ]
    )

    assert exit_code == 0
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_spec"] == "other/repo:Q8_0"
