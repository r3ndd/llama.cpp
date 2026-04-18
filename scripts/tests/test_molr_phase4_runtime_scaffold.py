from __future__ import annotations

import json

import numpy as np
import pytest

from molr.package_molr_bundle import main as package_main
from molr.runtime_bundle import MolrRuntimeBundleError, load_runtime_bundle
from molr.runtime_shadow import main as runtime_shadow_main
from molr.runtime_telemetry import main as runtime_telemetry_main
from molr.types import (
    MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
    MOLR_RUNTIME_CONFIG_SCHEMA_VERSION,
    MOLR_RUNTIME_SHADOW_REPORT_SCHEMA_VERSION,
    MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION,
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


def test_package_bundle_can_emit_runtime_template_and_load_runtime_state(tmp_path) -> None:
    checkpoints_dir = tmp_path / "phase2" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    plan_path = tmp_path / "molr_plan.json"
    thresholds_path = tmp_path / "molr_thresholds.json"
    bundle_dir = tmp_path / "bundle"

    _write_plan(plan_path, experts=[(0, 0)])
    _write_checkpoint(checkpoints_dir / "molr_expert_0_0.npz", layer=0, expert=0)

    thresholds_payload = {
        "schema_version": MOLR_THRESHOLDS_SCHEMA_VERSION,
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "lookup_table": [{"threshold": 0.0, "fallback_rate": 1.0, "quality_proxy": 1.0}],
        "quality_profiles": {
            "balanced": {
                "quality_proxy_min": 0.9,
                "selected_threshold": 0.42,
                "selected_fallback_rate": 0.2,
                "selected_quality_proxy": 0.95,
            }
        },
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
            "--emit-runtime-config-template",
            "--out-dir",
            str(bundle_dir),
        ]
    )
    assert exit_code == 0

    runtime_cfg_path = bundle_dir / "runtime_config.json"
    runtime_cfg_path.write_text(
        json.dumps(
            {
                "schema_version": MOLR_RUNTIME_CONFIG_SCHEMA_VERSION,
                "enabled": True,
                "quality_profile": "balanced",
                "telemetry_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    runtime_state = load_runtime_bundle(
        bundle_dir=bundle_dir,
        runtime_config_path=runtime_cfg_path,
        expected_model="unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        require_explicit_enable=True,
    )

    assert runtime_state["runtime"]["enabled"] is True
    assert runtime_state["runtime"]["fallback_threshold_source"] == "quality_profile:balanced"
    assert runtime_state["runtime"]["fallback_threshold"] == 0.42
    assert len(runtime_state["checkpoints"]) == 1

    runtime_template = json.loads((bundle_dir / "runtime_config.template.json").read_text(encoding="utf-8"))
    assert runtime_template["enabled"] is False


def test_runtime_telemetry_and_shadow_harness_emit_phase4_reports(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    telemetry_path = tmp_path / "telemetry.json"
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "layer": 0,
                        "expert": 0,
                        "used_fallback": False,
                        "predicted_error": 0.2,
                        "molr_latency_ms": 1.0,
                        "fallback_latency_ms": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "layer": 0,
                        "expert": 0,
                        "used_fallback": True,
                        "predicted_error": 0.8,
                        "molr_latency_ms": 1.1,
                        "fallback_latency_ms": 4.5,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    exit_code = runtime_telemetry_main(
        [
            "--events-jsonl",
            str(events_path),
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--out-json",
            str(telemetry_path),
        ]
    )
    assert exit_code == 0

    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["schema_version"] == MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION
    assert telemetry["window"]["events_total"] == 2
    assert telemetry["experts"][0]["fallback_calls_total"] == 1

    checkpoints_dir = tmp_path / "phase2" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plan_path = tmp_path / "molr_plan.json"
    thresholds_path = tmp_path / "molr_thresholds.json"
    bundle_dir = tmp_path / "bundle"

    _write_plan(plan_path, experts=[(0, 0)])
    _write_checkpoint(checkpoints_dir / "molr_expert_0_0.npz", layer=0, expert=0)
    thresholds_payload = {
        "schema_version": MOLR_THRESHOLDS_SCHEMA_VERSION,
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "lookup_table": [{"threshold": 0.0, "fallback_rate": 1.0, "quality_proxy": 1.0}],
        "quality_profiles": {
            "balanced": {
                "quality_proxy_min": 0.9,
                "selected_threshold": 0.3,
                "selected_fallback_rate": 0.2,
                "selected_quality_proxy": 0.95,
            }
        },
    }
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")
    package_exit = package_main(
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
    assert package_exit == 0

    runtime_cfg_path = tmp_path / "runtime_config.json"
    runtime_cfg_path.write_text(
        json.dumps(
            {
                "schema_version": MOLR_RUNTIME_CONFIG_SCHEMA_VERSION,
                "enabled": True,
                "quality_profile": "balanced",
                "telemetry_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    shadow_out = tmp_path / "runtime_shadow_report.json"
    shadow_exit = runtime_shadow_main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--runtime-config-json",
            str(runtime_cfg_path),
            "--telemetry-json",
            str(telemetry_path),
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--fallback-rate-alert-threshold",
            "0.4",
            "--latency-ratio-alert-threshold",
            "2.0",
            "--require-explicit-enable",
            "--out-json",
            str(shadow_out),
        ]
    )
    assert shadow_exit == 0

    shadow = json.loads(shadow_out.read_text(encoding="utf-8"))
    assert shadow["schema_version"] == MOLR_RUNTIME_SHADOW_REPORT_SCHEMA_VERSION
    assert shadow["runtime"]["enabled"] is True
    assert shadow["telemetry_summary"]["totals"]["fallback_calls_total"] == 1
    assert len(shadow["telemetry_summary"]["alerts"]) >= 1


def test_runtime_bundle_rejects_config_with_both_threshold_and_profile(tmp_path) -> None:
    checkpoints_dir = tmp_path / "phase2" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    plan_path = tmp_path / "molr_plan.json"
    thresholds_path = tmp_path / "molr_thresholds.json"
    bundle_dir = tmp_path / "bundle"

    _write_plan(plan_path, experts=[(0, 0)])
    _write_checkpoint(checkpoints_dir / "molr_expert_0_0.npz", layer=0, expert=0)

    thresholds_payload = {
        "schema_version": MOLR_THRESHOLDS_SCHEMA_VERSION,
        "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "lookup_table": [{"threshold": 0.0, "fallback_rate": 1.0, "quality_proxy": 1.0}],
        "quality_profiles": {
            "balanced": {
                "quality_proxy_min": 0.9,
                "selected_threshold": 0.42,
                "selected_fallback_rate": 0.2,
                "selected_quality_proxy": 0.95,
            }
        },
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

    runtime_cfg_path = tmp_path / "runtime_config.json"
    runtime_cfg_path.write_text(
        json.dumps(
            {
                "schema_version": MOLR_RUNTIME_CONFIG_SCHEMA_VERSION,
                "enabled": True,
                "quality_profile": "balanced",
                "fallback_threshold": 0.1,
                "telemetry_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MolrRuntimeBundleError, match="exactly one"):
        load_runtime_bundle(
            bundle_dir=bundle_dir,
            runtime_config_path=runtime_cfg_path,
            expected_model="unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            require_explicit_enable=True,
        )
