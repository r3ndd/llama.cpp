from __future__ import annotations

import json

import numpy as np

from molr.capture_expert_covariance import main as covariance_main
from molr.plan_from_svd import main as plan_main
from molr.types import (
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
    MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
)


def _base_svd_report() -> dict:
    return {
        "schema_version": "1.1",
        "run": {
            "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "fidelity_mode": "full_svd",
            "git_commit": "abc123",
        },
        "per_matrix": [
            {
                "tensor": "blk.0.ffn_gate.0",
                "source_tensor": "blk.0.ffn_gate.weight",
                "layer": 0,
                "expert": 0,
                "role": "gate",
                "shape": [8, 4],
                "singular_value_count": 4,
                "fro_norm": 1.0,
                "explained_spectral_energy_rank_fractions": [
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                    0.50,
                    0.60,
                    0.70,
                    0.80,
                    0.90,
                    0.91,
                    0.92,
                    0.93,
                    0.94,
                    0.95,
                    0.96,
                    0.97,
                    0.98,
                    0.99,
                    1.0,
                ],
            },
            {
                "tensor": "blk.0.ffn_up.0",
                "source_tensor": "blk.0.ffn_up.weight",
                "layer": 0,
                "expert": 0,
                "role": "up",
                "shape": [8, 4],
                "singular_value_count": 4,
                "fro_norm": 2.0,
                "explained_spectral_energy_rank_fractions": [
                    0.05,
                    0.10,
                    0.15,
                    0.20,
                    0.25,
                    0.30,
                    0.35,
                    0.40,
                    0.45,
                    0.50,
                    0.55,
                    0.60,
                    0.65,
                    0.70,
                    0.75,
                    0.80,
                    0.84,
                    0.88,
                    0.89,
                ],
            },
        ],
    }


def test_plan_from_svd_writes_plan_with_rank_policy_and_partitions(tmp_path) -> None:
    svd_path = tmp_path / "svd_report.json"
    out_path = tmp_path / "molr_plan.json"
    svd_path.write_text(json.dumps(_base_svd_report()), encoding="utf-8")

    exit_code = plan_main(
        [
            "--svd-json",
            str(svd_path),
            "--target-energy",
            "0.90",
            "--k-components",
            "4",
            "--out-json",
            str(out_path),
        ]
    )
    assert exit_code == 0

    plan = json.loads(out_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == MOLR_PLAN_SCHEMA_VERSION
    assert plan["target_energy"] == 0.90
    assert plan["default_k"] == 4
    assert plan["summary"]["experts_total"] == 1
    assert plan["summary"]["matrices_total"] == 2

    expert = plan["experts"][0]
    gate = next(m for m in expert["matrices"] if m["role"] == "gate")
    assert gate["rank"] == 2
    assert gate["target_energy_met_on_fraction_grid"] is True
    assert gate["init_partition"]["strategy"] == "strided"
    assert len(gate["init_partition"]["component_assignments"]) == 4


def test_plan_from_svd_clamps_to_full_rank_when_grid_cannot_meet_target(tmp_path) -> None:
    svd_path = tmp_path / "svd_report.json"
    out_path = tmp_path / "molr_plan.json"
    svd_path.write_text(json.dumps(_base_svd_report()), encoding="utf-8")

    exit_code = plan_main(
        [
            "--svd-json",
            str(svd_path),
            "--target-energy",
            "0.95",
            "--k-components",
            "4",
            "--out-json",
            str(out_path),
        ]
    )
    assert exit_code == 0

    plan = json.loads(out_path.read_text(encoding="utf-8"))
    up = next(m for m in plan["experts"][0]["matrices"] if m["role"] == "up")
    assert up["target_energy_met_on_fraction_grid"] is False
    assert up["selection_notes"]["clamped_to_full_rank"] is True
    assert up["rank"] == up["min_dimension"]


def test_covariance_scaffold_allow_empty_writes_empty_contract_outputs(tmp_path) -> None:
    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

    exit_code = covariance_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--tokens",
            "0",
            "--allow-empty",
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["schema_version"] == MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION
    assert summary["status"] == "empty"

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION
    assert payload["mu"].shape == (0, 0)
    assert payload["chol"].shape == (0, 0, 0)


def test_covariance_scaffold_contract_computes_success_and_failure_accounting(tmp_path) -> None:
    routed_path = tmp_path / "routed_inputs.npz"
    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

    # Expert (0,0): 3 samples -> success with min-samples=3
    # Expert (0,1): 2 samples -> explicit insufficient-samples failure
    inputs = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [2.0, 1.0],
            [2.0, 1.2],
        ],
        dtype=np.float32,
    )
    layers = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    experts = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    np.savez(routed_path, inputs=inputs, layers=layers, experts=experts)

    exit_code = covariance_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--tokens",
            "5",
            "--routed-inputs-npz",
            str(routed_path),
            "--min-samples-per-expert",
            "3",
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["schema_version"] == MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION
    assert summary["failure_accounting"]["experts_succeeded_total"] == 1
    assert summary["failure_accounting"]["experts_failed_total"] == 1
    assert summary["failure_accounting"]["by_reason"]["insufficient_samples(<3)"] == 1

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION
    assert payload["d_model"].item() == 2
    assert payload["layers"].shape == (1,)
    assert payload["experts"].shape == (1,)
    assert payload["mu"].shape == (1, 2)
    assert payload["chol"].shape == (1, 2, 2)


def test_covariance_scaffold_reports_observed_vs_processed_expert_counts(tmp_path) -> None:
    routed_path = tmp_path / "routed_inputs.npz"
    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

    inputs = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.3],
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.2],
        ],
        dtype=np.float32,
    )
    layers = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    experts = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    np.savez(routed_path, inputs=inputs, layers=layers, experts=experts)

    exit_code = covariance_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--tokens",
            "6",
            "--routed-inputs-npz",
            str(routed_path),
            "--min-samples-per-expert",
            "3",
            "--max-experts",
            "1",
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["observed"]["experts_observed_total"] == 2
    assert summary["observed"]["experts_processed_total"] == 1
