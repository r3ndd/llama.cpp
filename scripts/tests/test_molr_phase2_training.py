from __future__ import annotations

import json

import numpy as np

from molr.train_all_experts import main as train_all_main
from molr.train_expert_molr import compute_objective_and_gradients, main as train_expert_main
from molr.types import (
    MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
    MOLR_EXPERT_VALIDATION_SCHEMA_VERSION,
    MOLR_EXPERT_WEIGHTS_NPZ_SCHEMA_VERSION,
    MOLR_FAILURE_LEDGER_SCHEMA_VERSION,
    MOLR_VALIDATION_REPORT_SCHEMA_VERSION,
)


def _write_plan(path, *, experts: list[tuple[int, int]], k: int = 2, rank: int = 2) -> None:
    component_assignments = [
        {"component": 0, "rank_indices": [0]},
        {"component": 1, "rank_indices": [1]},
    ]
    payload = {
        "schema_version": "molr_plan.v1",
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "default_k": k,
        "experts": [],
    }
    for layer, expert in experts:
        payload["experts"].append(
            {
                "layer": layer,
                "expert": expert,
                "matrices": [
                    {
                        "role": "gate",
                        "rank": rank,
                        "k_components": k,
                        "init_partition": {"component_assignments": component_assignments},
                    },
                    {
                        "role": "up",
                        "rank": rank,
                        "k_components": k,
                        "init_partition": {"component_assignments": component_assignments},
                    },
                    {
                        "role": "down",
                        "rank": rank,
                        "k_components": k,
                        "init_partition": {"component_assignments": component_assignments},
                    },
                ],
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase2_objective_keeps_error_supervision_detached() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 3))
    y_true = rng.normal(size=(4, 3))
    params = {
        "router_w": rng.normal(size=(3, 2)),
        "router_b": rng.normal(size=(2,)),
        "error_w": rng.normal(size=(3,)),
        "error_b": np.array(0.1, dtype=np.float64),
        "components": [
            {
                "gate_A": rng.normal(size=(3, 2)),
                "gate_B": rng.normal(size=(2, 4)),
                "up_A": rng.normal(size=(3, 2)),
                "up_B": rng.normal(size=(2, 4)),
                "down_A": rng.normal(size=(4, 2)),
                "down_B": rng.normal(size=(2, 3)),
            },
            {
                "gate_A": rng.normal(size=(3, 2)),
                "gate_B": rng.normal(size=(2, 4)),
                "up_A": rng.normal(size=(3, 2)),
                "up_B": rng.normal(size=(2, 4)),
                "down_A": rng.normal(size=(4, 2)),
                "down_B": rng.normal(size=(2, 3)),
            },
        ],
    }

    _, grads_no_err, _ = compute_objective_and_gradients(
        x=x,
        y_true=y_true,
        params=params,
        lambda_lb=0.01,
        lambda_err=0.0,
        detach_true_error_target=True,
    )
    _, grads_with_err, _ = compute_objective_and_gradients(
        x=x,
        y_true=y_true,
        params=params,
        lambda_lb=0.01,
        lambda_err=0.1,
        detach_true_error_target=True,
    )

    assert np.allclose(grads_no_err["router_w"], grads_with_err["router_w"])
    assert np.allclose(grads_no_err["router_b"], grads_with_err["router_b"])
    for k in range(2):
        for name in ("gate_A", "gate_B", "up_A", "up_B", "down_A", "down_B"):
            assert np.allclose(grads_no_err["components"][k][name], grads_with_err["components"][k][name])

    assert not np.allclose(grads_no_err["error_w"], grads_with_err["error_w"])


def test_train_expert_molr_writes_checkpoint_and_validation_contracts(tmp_path) -> None:
    plan_path = tmp_path / "molr_plan.json"
    cov_path = tmp_path / "covariance_stats.npz"
    weights_path = tmp_path / "expert_weights_0_0.npz"
    out_ckpt = tmp_path / "molr_expert_0_0.npz"
    out_val = tmp_path / "molr_validation_0_0.json"

    _write_plan(plan_path, experts=[(0, 0)], k=2, rank=2)

    d_model = 3
    np.savez(
        cov_path,
        schema_version=np.array("molr_covariance_npz.v1"),
        model_spec=np.array("unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"),
        d_model=np.array(d_model, dtype=np.int64),
        layers=np.array([0], dtype=np.int64),
        experts=np.array([0], dtype=np.int64),
        sample_count=np.array([32], dtype=np.int64),
        jitter_used=np.array([0.0], dtype=np.float64),
        mu=np.zeros((1, d_model), dtype=np.float32),
        chol=np.stack([np.eye(d_model, dtype=np.float32)], axis=0),
    )

    np.savez(
        weights_path,
        schema_version=np.array(MOLR_EXPERT_WEIGHTS_NPZ_SCHEMA_VERSION),
        gate=np.array(
            [
                [0.2, -0.1, 0.3, 0.0],
                [0.1, 0.4, -0.2, 0.5],
                [0.0, -0.3, 0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        up=np.array(
            [
                [0.1, 0.0, 0.3, -0.2],
                [0.2, -0.1, 0.0, 0.4],
                [-0.3, 0.2, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
        down=np.array(
            [
                [0.3, -0.1, 0.2],
                [0.1, 0.2, -0.2],
                [0.0, -0.1, 0.4],
                [0.2, 0.0, 0.1],
            ],
            dtype=np.float32,
        ),
    )

    exit_code = train_expert_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--plan-json",
            str(plan_path),
            "--cov-npz",
            str(cov_path),
            "--weights-npz",
            str(weights_path),
            "--layer",
            "0",
            "--expert",
            "0",
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--validation-samples",
            "8",
            "--log-interval",
            "0",
            "--out-checkpoint",
            str(out_ckpt),
            "--out-validation",
            str(out_val),
        ]
    )
    assert exit_code == 0

    validation = json.loads(out_val.read_text(encoding="utf-8"))
    assert validation["schema_version"] == MOLR_EXPERT_VALIDATION_SCHEMA_VERSION
    assert validation["layer"] == 0
    assert validation["expert"] == 0
    assert "cosine_similarity_mean" in validation["validation_metrics"]
    assert "router_entropy_mean" in validation["validation_metrics"]
    assert "error_head_pearson_r" in validation["validation_metrics"]

    ckpt = np.load(out_ckpt, allow_pickle=False)
    assert str(ckpt["schema_version"]) == MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION
    assert ckpt["router_w"].shape == (3, 2)
    assert "component_0_gate_A" in ckpt
    assert "component_1_down_B" in ckpt


def test_train_all_experts_emits_merged_report_and_failure_ledger(tmp_path) -> None:
    plan_path = tmp_path / "molr_plan.json"
    cov_path = tmp_path / "covariance_stats.npz"
    weights_dir = tmp_path / "expert_weights"
    out_dir = tmp_path / "phase2"
    weights_dir.mkdir(parents=True, exist_ok=True)

    _write_plan(plan_path, experts=[(0, 0), (0, 1)], k=2, rank=2)
    np.savez(
        cov_path,
        schema_version=np.array("molr_covariance_npz.v1"),
        model_spec=np.array("unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"),
        d_model=np.array(3, dtype=np.int64),
        layers=np.array([0], dtype=np.int64),
        experts=np.array([0], dtype=np.int64),
        sample_count=np.array([32], dtype=np.int64),
        jitter_used=np.array([0.0], dtype=np.float64),
        mu=np.zeros((1, 3), dtype=np.float32),
        chol=np.stack([np.eye(3, dtype=np.float32)], axis=0),
    )

    exit_code = train_all_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--plan-json",
            str(plan_path),
            "--cov-npz",
            str(cov_path),
            "--weights-dir",
            str(weights_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    report = json.loads((out_dir / "molr_validation_report.json").read_text(encoding="utf-8"))
    ledger = json.loads((out_dir / "molr_failure_ledger.json").read_text(encoding="utf-8"))

    assert report["schema_version"] == MOLR_VALIDATION_REPORT_SCHEMA_VERSION
    assert report["summary"]["experts_trained_total"] == 0
    assert report["summary"]["experts_orchestration_fail_total"] == 2

    assert ledger["schema_version"] == MOLR_FAILURE_LEDGER_SCHEMA_VERSION
    assert ledger["summary"]["entries_total"] == 2
    statuses = {entry["status"] for entry in ledger["entries"]}
    assert "train_skipped_missing_weights" in statuses
    assert "train_skipped_missing_cov" in statuses
