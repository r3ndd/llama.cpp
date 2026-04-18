from __future__ import annotations

import json

import numpy as np

from molr.calibrate_fallback import main as calibrate_main
from molr.package_molr_bundle import main as package_main
from molr.types import (
    MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION,
    MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
    MOLR_EXPERT_VALIDATION_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
    MOLR_THRESHOLDS_SCHEMA_VERSION,
)


def _write_plan(path, experts: list[tuple[int, int]]) -> None:
    payload = {
        "schema_version": MOLR_PLAN_SCHEMA_VERSION,
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "default_k": 2,
        "experts": [{"layer": layer, "expert": expert, "matrices": []} for layer, expert in experts],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_validation(path, *, layer: int, expert: int, pred: float, true: float, status: str = "pass") -> None:
    payload = {
        "schema_version": MOLR_EXPERT_VALIDATION_SCHEMA_VERSION,
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "layer": layer,
        "expert": expert,
        "status": status,
        "failure_reasons": [] if status == "pass" else ["synthetic_failure"],
        "validation_metrics": {
            "pred_error_mean": pred,
            "true_error_mean": true,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_checkpoint(path, *, layer: int, expert: int) -> None:
    np.savez(
        path,
        schema_version=np.array(MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION),
        model_spec=np.array("unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"),
        layer=np.array(layer, dtype=np.int64),
        expert=np.array(expert, dtype=np.int64),
        k_components=np.array(2, dtype=np.int64),
        d_model=np.array(3, dtype=np.int64),
        d_intermediate=np.array(4, dtype=np.int64),
        router_w=np.zeros((3, 2), dtype=np.float32),
        router_b=np.zeros((2,), dtype=np.float32),
        error_w=np.zeros((3,), dtype=np.float32),
        error_b=np.array(0.0, dtype=np.float32),
        component_0_gate_A=np.zeros((3, 2), dtype=np.float32),
        component_0_gate_B=np.zeros((2, 4), dtype=np.float32),
        component_0_up_A=np.zeros((3, 2), dtype=np.float32),
        component_0_up_B=np.zeros((2, 4), dtype=np.float32),
        component_0_down_A=np.zeros((4, 2), dtype=np.float32),
        component_0_down_B=np.zeros((2, 3), dtype=np.float32),
        component_1_gate_A=np.zeros((3, 2), dtype=np.float32),
        component_1_gate_B=np.zeros((2, 4), dtype=np.float32),
        component_1_up_A=np.zeros((3, 2), dtype=np.float32),
        component_1_up_B=np.zeros((2, 4), dtype=np.float32),
        component_1_down_A=np.zeros((4, 2), dtype=np.float32),
        component_1_down_B=np.zeros((2, 3), dtype=np.float32),
    )


def test_calibrate_fallback_emits_threshold_lookup_and_profiles(tmp_path) -> None:
    phase2_dir = tmp_path / "phase2"
    checkpoints_dir = phase2_dir / "checkpoints"
    validation_dir = phase2_dir / "validation"
    out_json = tmp_path / "molr_thresholds.json"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    _write_validation(validation_dir / "molr_validation_0_0.json", layer=0, expert=0, pred=0.2, true=0.1)
    _write_validation(validation_dir / "molr_validation_0_1.json", layer=0, expert=1, pred=0.8, true=0.6)
    _write_validation(
        validation_dir / "molr_validation_0_2.json",
        layer=0,
        expert=2,
        pred=0.5,
        true=0.4,
        status="fail",
    )

    exit_code = calibrate_main(
        [
            "--checkpoints",
            str(checkpoints_dir),
            "--validation-dir",
            str(validation_dir),
            "--quality-profiles",
            "balanced:0.70,strict:0.85",
            "--top-cache-candidates",
            "2",
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MOLR_THRESHOLDS_SCHEMA_VERSION
    assert payload["summary"]["experts_total"] == 3
    assert len(payload["lookup_table"]) >= 2
    assert "balanced" in payload["quality_profiles"]
    assert "strict" in payload["quality_profiles"]
    assert len(payload["cache_candidates"]) == 2
    assert payload["cache_candidates"][0]["status"] == "fail"


def test_package_molr_bundle_emits_manifest_with_checksums_and_coverage(tmp_path) -> None:
    phase2_dir = tmp_path / "phase2"
    checkpoints_dir = phase2_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    plan_path = tmp_path / "molr_plan.json"
    thresholds_path = tmp_path / "molr_thresholds.json"
    bundle_dir = tmp_path / "bundle"

    _write_plan(plan_path, experts=[(0, 0), (0, 1)])
    _write_checkpoint(checkpoints_dir / "molr_expert_0_0.npz", layer=0, expert=0)

    thresholds_payload = {
        "schema_version": MOLR_THRESHOLDS_SCHEMA_VERSION,
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "lookup_table": [{"threshold": 0.0, "fallback_rate": 1.0, "quality_proxy": 1.0}],
        "quality_profiles": {"balanced": {"selected_threshold": 0.0}},
    }
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")

    exit_code = package_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--plan-json",
            str(plan_path),
            "--checkpoints",
            str(checkpoints_dir),
            "--thresholds-json",
            str(thresholds_path),
            "--out-dir",
            str(bundle_dir),
        ]
    )
    assert exit_code == 0

    manifest = json.loads((bundle_dir / "molr_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION
    assert manifest["compatibility"]["plan_schema_version"] == MOLR_PLAN_SCHEMA_VERSION
    assert manifest["compatibility"]["thresholds_schema_version"] == MOLR_THRESHOLDS_SCHEMA_VERSION
    assert manifest["coverage"]["plan_experts_total"] == 2
    assert manifest["coverage"]["checkpoint_experts_total"] == 1
    assert manifest["coverage"]["missing_plan_experts_total"] == 1
    assert len(manifest["artifacts"]["checkpoints"]) == 1
