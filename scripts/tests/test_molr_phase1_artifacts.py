from __future__ import annotations

import json
import subprocess

import numpy as np

from molr.capture_expert_covariance import main as covariance_main
from molr.plan_from_svd import main as plan_main
from molr.types import (
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2,
    MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION,
    MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION_V2,
    MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION,
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
    assert summary["schema_version"] == MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION_V2
    assert summary["status"] == "empty"
    assert summary["input_contract"]["mode"] == "contract_only"

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2
    assert str(payload["granularity"]) == "layer"
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
    assert summary["schema_version"] == MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION_V2
    assert summary["failure_accounting"]["experts_succeeded_total"] == 1
    assert summary["failure_accounting"]["experts_failed_total"] == 0
    assert summary["experts_fallback_used_total"] == 0
    assert summary["experts_fallback_failed_total"] == 0

    succeeded = summary["experts_succeeded"][0]
    assert succeeded["sample_source"] == "layer"
    assert succeeded["effective_sample_count"] == 5

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2
    assert str(payload["granularity"]) == "layer"
    assert payload["d_model"].item() == 2
    assert payload["layers"].shape == (1,)
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
    assert summary["observed"]["experts_processed_total"] is None
    assert summary["observed"]["layers_processed_total"] == 1


def test_covariance_scaffold_ignores_expert_fallback_flags_in_layer_mode(tmp_path) -> None:
    routed_path = tmp_path / "routed_inputs.npz"
    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

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
            "--fallback-to-layer-inputs-on-low-samples",
            "--min-layer-samples-for-fallback",
            "4",
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["failure_accounting"]["experts_succeeded_total"] == 1
    assert summary["failure_accounting"]["experts_failed_total"] == 0
    assert summary["experts_fallback_used_total"] == 0
    assert summary["experts_fallback_failed_total"] == 0
    assert summary["deprecation_warnings"]
    assert summary["experts_succeeded"][0]["sample_source"] == "layer"
    assert summary["experts_succeeded"][0]["effective_sample_count"] == 5

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2
    assert str(payload["granularity"]) == "layer"
    assert payload["layers"].shape == (1,)
    assert payload["sample_count"].tolist() == [5]


def test_covariance_scaffold_min_samples_per_layer_enforced(tmp_path) -> None:
    routed_path = tmp_path / "routed_inputs.npz"
    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

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
            "--min-samples-per-layer",
            "6",
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["failure_accounting"]["experts_succeeded_total"] == 0
    assert summary["failure_accounting"]["experts_failed_total"] == 1
    assert summary["failure_accounting"]["by_reason"]["insufficient_layer_samples(<6)"] == 1
    assert summary["experts_fallback_used_total"] == 0
    assert summary["experts_fallback_failed_total"] == 0


def test_covariance_scaffold_capture_mode_with_optional_layer_trace_output(tmp_path) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        "\n".join(
            [
                '{"prompt": "p0", "inference_params": {"n_predict": 1}}',
                '{"prompt": "p1", "inference_params": {"n_predict": 1}}',
                '{"prompt": "p2", "inference_params": {"n_predict": 1}}',
                '{"prompt": "p3", "inference_params": {"n_predict": 1}}',
                '{"prompt": "p4", "inference_params": {"n_predict": 1}}',
            ]
        ),
        encoding="utf-8",
    )

    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"
    out_traces = tmp_path / "layer_traces.npz"
    trace_jsonl = tmp_path / "bridge_trace.jsonl"

    prompt_to_trace = {
        "p0": {"inputs": [1.0, 0.0], "layer": 0, "expert": 0},
        "p1": {"inputs": [0.5, 0.5], "layer": 0, "expert": 0},
        "p2": {"inputs": [0.0, 1.0], "layer": 0, "expert": 0},
        "p3": {"inputs": [2.0, 1.0], "layer": 0, "expert": 1},
        "p4": {"inputs": [2.0, 1.2], "layer": 0, "expert": 1},
    }

    def _fake_run(cmd, env=None, **_kwargs):
        assert env is not None
        assert env["LLAMA_MOE_TRACE_ENABLE"] == "1"
        assert env["LLAMA_MOE_TRACE_FORMAT"] == "jsonl"
        assert env["LLAMA_MOLR_CAPTURE_SOURCE"]
        assert "--no-display-prompt" in cmd
        trace_path = env["LLAMA_MOE_TRACE_JSONL"]
        prompt = cmd[cmd.index("-p") + 1]
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(prompt_to_trace[prompt]) + "\n")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    import molr.capture_expert_covariance as cov_mod

    original_run = cov_mod.subprocess.run
    cov_mod.subprocess.run = _fake_run
    try:
        exit_code = covariance_main(
            [
                "--model",
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
                "--tokens",
                "5",
                "--capture-routed-traces",
                "--capture-prompts-jsonl",
                str(prompts_path),
                "--capture-trace-jsonl",
                str(trace_jsonl),
                "--min-samples-per-expert",
                "3",
                "--fallback-to-layer-inputs-on-low-samples",
                "--out-layer-traces-npz",
                str(out_traces),
                "--trace-dtype",
                "float32",
                "--out-npz",
                str(out_npz),
                "--out-json",
                str(out_json),
            ]
        )
    finally:
        cov_mod.subprocess.run = original_run
    assert exit_code == 0

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["capture_runtime"]["status"] == "capture_enabled"
    assert summary["input_contract"]["mode"] == "capture_enabled"
    assert summary["capture_runtime"]["trace_jsonl_path"] == str(trace_jsonl.resolve())
    assert summary["capture_runtime"]["prompt_records_total"] == 5
    assert summary["outputs"]["layer_traces_npz"] == str(out_traces.resolve())
    assert summary["outputs"]["layer_traces_npz_schema_version"] == MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION
    assert summary["outputs"]["routed_traces_npz"] is None
    assert summary["deprecation_warnings"]

    traces = np.load(out_traces, allow_pickle=False)
    assert str(traces["schema_version"]) == MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION
    assert traces["inputs"].shape == (5, 2)
    assert traces["layers"].shape == (5,)


def test_covariance_scaffold_capture_flag_validation_matrix(tmp_path) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"prompt": "p0", "inference_params": {"n_predict": 1}}\n', encoding="utf-8")
    routed_path = tmp_path / "routed_inputs.npz"
    np.savez(
        routed_path,
        inputs=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layers=np.array([0, 0], dtype=np.int64),
        experts=np.array([0, 0], dtype=np.int64),
    )

    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

    # capture enabled but missing --capture-prompts-jsonl
    try:
        covariance_main(
            [
                "--model",
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
                "--capture-routed-traces",
                "--out-npz",
                str(out_npz),
                "--out-json",
                str(out_json),
            ]
        )
        assert False, "expected argparse failure"
    except SystemExit as exc:
        assert exc.code == 2

    # capture and routed-inputs are mutually exclusive
    try:
        covariance_main(
            [
                "--model",
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
                "--capture-routed-traces",
                "--capture-prompts-jsonl",
                str(prompts_path),
                "--routed-inputs-npz",
                str(routed_path),
                "--out-npz",
                str(out_npz),
                "--out-json",
                str(out_json),
            ]
        )
        assert False, "expected argparse failure"
    except SystemExit as exc:
        assert exc.code == 2

    # --capture-prompts-jsonl without capture flag
    try:
        covariance_main(
            [
                "--model",
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
                "--capture-prompts-jsonl",
                str(prompts_path),
                "--out-npz",
                str(out_npz),
                "--out-json",
                str(out_json),
            ]
        )
        assert False, "expected argparse failure"
    except SystemExit as exc:
        assert exc.code == 2


def test_covariance_scaffold_capture_accepts_json_records_format(tmp_path) -> None:
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {
                "schema": "capture_prompt_inference.v1",
                "records": [
                    {"prompt": "p0", "inference_params": {"n_predict": 1}},
                    {"prompt": "p1", "inference_params": {"n_predict": 1}},
                    {"prompt": "p2", "inference_params": {"n_predict": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"
    trace_jsonl = tmp_path / "bridge_trace.jsonl"

    prompt_to_trace = {
        "p0": {"inputs": [1.0, 0.0], "layer": 0, "expert": 0},
        "p1": {"inputs": [0.5, 0.5], "layer": 0, "expert": 0},
        "p2": {"inputs": [0.0, 1.0], "layer": 0, "expert": 0},
    }

    def _fake_run(cmd, env=None, **_kwargs):
        prompt = cmd[cmd.index("-p") + 1]
        with open(env["LLAMA_MOE_TRACE_JSONL"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(prompt_to_trace[prompt]) + "\n")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    import molr.capture_expert_covariance as cov_mod

    original_run = cov_mod.subprocess.run
    cov_mod.subprocess.run = _fake_run
    try:
        exit_code = covariance_main(
            [
                "--model",
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
                "--tokens",
                "3",
                "--capture-routed-traces",
                "--capture-prompts-jsonl",
                str(prompts_path),
                "--capture-trace-jsonl",
                str(trace_jsonl),
                "--min-samples-per-expert",
                "3",
                "--out-npz",
                str(out_npz),
                "--out-json",
                str(out_json),
            ]
        )
    finally:
        cov_mod.subprocess.run = original_run

    assert exit_code == 0
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["capture_runtime"]["prompt_inference_source_schema"] == "capture_prompt_inference.v1"
    assert summary["capture_runtime"]["status"] == "capture_enabled"


def test_covariance_scaffold_capture_rejects_invalid_prompt_record_schema(tmp_path) -> None:
    prompts_path = tmp_path / "bad_prompts.jsonl"
    prompts_path.write_text('{"inference_params": {"n_predict": 1}}\n', encoding="utf-8")

    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

    exit_code = covariance_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--tokens",
            "1",
            "--capture-routed-traces",
            "--capture-prompts-jsonl",
            str(prompts_path),
            "--capture-trace-jsonl",
            str(tmp_path / "trace.jsonl"),
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 2


def test_covariance_scaffold_expert_granularity_preserves_v1_covariance_output(tmp_path) -> None:
    routed_path = tmp_path / "routed_inputs.npz"
    out_npz = tmp_path / "covariance_stats.npz"
    out_json = tmp_path / "covariance_summary.json"

    inputs = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    layers = np.array([0, 0, 0], dtype=np.int64)
    experts = np.array([0, 0, 0], dtype=np.int64)
    np.savez(routed_path, inputs=inputs, layers=layers, experts=experts)

    exit_code = covariance_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--tokens",
            "3",
            "--input-granularity",
            "expert",
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

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION
    assert payload["experts"].shape == (1,)


def test_covariance_scaffold_layer_granularity_accepts_contract_without_experts_array(tmp_path) -> None:
    routed_path = tmp_path / "routed_inputs_layer_only.npz"
    out_npz = tmp_path / "covariance_stats_layer.npz"
    out_json = tmp_path / "covariance_summary_layer.json"

    np.savez(
        routed_path,
        inputs=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        layers=np.array([0, 0, 0], dtype=np.int64),
    )

    exit_code = covariance_main(
        [
            "--model",
            "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
            "--tokens",
            "3",
            "--input-granularity",
            "layer",
            "--routed-inputs-npz",
            str(routed_path),
            "--min-samples-per-layer",
            "3",
            "--out-npz",
            str(out_npz),
            "--out-json",
            str(out_json),
        ]
    )
    assert exit_code == 0

    payload = np.load(out_npz, allow_pickle=False)
    assert str(payload["schema_version"]) == MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2
    assert str(payload["granularity"]) == "layer"
    assert "experts" not in payload.files
