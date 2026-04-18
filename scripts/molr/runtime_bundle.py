from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION,
    MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
    MOLR_RUNTIME_CONFIG_SCHEMA_VERSION,
    MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION,
    MOLR_THRESHOLDS_SCHEMA_VERSION,
)


class MolrRuntimeBundleError(RuntimeError):
    """Raised when a MoLR runtime bundle cannot be validated/loaded safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrRuntimeBundleError(f"Failed reading JSON '{path}': {exc}") from exc


def _npz_scalar_string(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key])
    if value.ndim == 0:
        return str(value.item())
    return str(value)


def _validate_manifest(manifest: dict[str, Any], expected_model: str | None) -> tuple[str, dict[str, Any]]:
    schema = str(manifest.get("schema_version") or "")
    if schema != MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            f"Manifest schema mismatch: got '{schema}', expected '{MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION}'",
        )

    model_spec = str(manifest.get("model_spec") or "")
    if not model_spec:
        raise MolrRuntimeBundleError("Manifest missing model_spec")
    if expected_model and model_spec != expected_model:
        raise MolrRuntimeBundleError(
            f"Manifest model_spec mismatch: got '{model_spec}', expected '{expected_model}'",
        )

    compatibility = manifest.get("compatibility", {})
    plan_schema = str(compatibility.get("plan_schema_version") or "")
    if plan_schema != MOLR_PLAN_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            f"Manifest plan schema mismatch: got '{plan_schema}', expected '{MOLR_PLAN_SCHEMA_VERSION}'",
        )
    thresholds_schema = str(compatibility.get("thresholds_schema_version") or "")
    if thresholds_schema != MOLR_THRESHOLDS_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            "Manifest thresholds schema mismatch: "
            f"got '{thresholds_schema}', expected '{MOLR_THRESHOLDS_SCHEMA_VERSION}'",
        )
    checkpoint_schema = str(compatibility.get("checkpoint_schema_version") or "")
    if checkpoint_schema != MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            "Manifest checkpoint schema mismatch: "
            f"got '{checkpoint_schema}', expected '{MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION}'",
        )

    runtime_config_schema = str(compatibility.get("runtime_config_schema_version") or "")
    if runtime_config_schema != MOLR_RUNTIME_CONFIG_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            "Manifest runtime-config schema mismatch: "
            f"got '{runtime_config_schema}', expected '{MOLR_RUNTIME_CONFIG_SCHEMA_VERSION}'",
        )

    runtime_telemetry_schema = str(compatibility.get("runtime_telemetry_schema_version") or "")
    if runtime_telemetry_schema != MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            "Manifest runtime-telemetry schema mismatch: "
            f"got '{runtime_telemetry_schema}', expected '{MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION}'",
        )

    return model_spec, compatibility


def _validate_runtime_config(
    config: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    require_explicit_enable: bool,
) -> dict[str, Any]:
    schema = str(config.get("schema_version") or "")
    if schema and schema != MOLR_RUNTIME_CONFIG_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            f"Runtime config schema mismatch: got '{schema}', expected '{MOLR_RUNTIME_CONFIG_SCHEMA_VERSION}'",
        )

    enabled = bool(config.get("enabled", False))
    if require_explicit_enable and not enabled:
        raise MolrRuntimeBundleError(
            "Runtime config disabled MoLR (enabled=false). "
            "Set enabled=true for explicit opt-in.",
        )

    explicit_threshold = config.get("fallback_threshold", None)
    profile_name = str(config.get("quality_profile") or "")
    if explicit_threshold is None and not profile_name:
        raise MolrRuntimeBundleError(
            "Runtime config must set either fallback_threshold or quality_profile",
        )
    if explicit_threshold is not None and profile_name:
        raise MolrRuntimeBundleError(
            "Runtime config must set exactly one of fallback_threshold or quality_profile",
        )

    resolved_threshold: float
    threshold_source: str
    if explicit_threshold is not None:
        try:
            resolved_threshold = float(explicit_threshold)
        except Exception as exc:
            raise MolrRuntimeBundleError("runtime fallback_threshold must be numeric") from exc
        threshold_source = "explicit"
    else:
        profiles = thresholds.get("quality_profiles", {})
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            raise MolrRuntimeBundleError(
                f"quality_profile '{profile_name}' missing from thresholds quality_profiles",
            )
        if "selected_threshold" not in profile:
            raise MolrRuntimeBundleError(
                f"quality_profile '{profile_name}' missing selected_threshold",
            )
        try:
            resolved_threshold = float(profile["selected_threshold"])
        except Exception as exc:
            raise MolrRuntimeBundleError(
                f"quality_profile '{profile_name}' selected_threshold is not numeric",
            ) from exc
        threshold_source = f"quality_profile:{profile_name}"

    if resolved_threshold < 0.0:
        raise MolrRuntimeBundleError(
            f"Resolved runtime threshold must be >= 0.0, got {resolved_threshold}",
        )

    return {
        "schema_version": MOLR_RUNTIME_CONFIG_SCHEMA_VERSION,
        "enabled": enabled,
        "quality_profile": profile_name,
        "fallback_threshold": resolved_threshold,
        "fallback_threshold_source": threshold_source,
        "telemetry_enabled": bool(config.get("telemetry_enabled", True)),
    }


def _validate_checkpoint_rows(bundle_root: Path, manifest: dict[str, Any], model_spec: str) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts", {})
    checkpoint_rows = artifacts.get("checkpoints", [])
    if not isinstance(checkpoint_rows, list) or not checkpoint_rows:
        raise MolrRuntimeBundleError("Manifest artifacts.checkpoints must be a non-empty list")

    out: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    for row in checkpoint_rows:
        if not isinstance(row, dict):
            raise MolrRuntimeBundleError("Manifest checkpoint row must be object")
        try:
            layer = int(row.get("layer"))
            expert = int(row.get("expert"))
        except Exception as exc:
            raise MolrRuntimeBundleError("Manifest checkpoint row has invalid layer/expert") from exc

        key = (layer, expert)
        if key in seen_keys:
            raise MolrRuntimeBundleError(f"Duplicate checkpoint entry for {key}")
        seen_keys.add(key)

        path_raw = str(row.get("path") or "")
        if not path_raw:
            raise MolrRuntimeBundleError(f"Manifest checkpoint row missing path for {key}")
        ckpt_path = Path(path_raw)
        if not ckpt_path.is_absolute():
            ckpt_path = (bundle_root / ckpt_path).resolve()
        if not ckpt_path.is_file():
            raise MolrRuntimeBundleError(f"Checkpoint file missing for {key}: '{ckpt_path}'")

        try:
            with np.load(ckpt_path, allow_pickle=False) as payload:
                schema = _npz_scalar_string(payload, "schema_version")
                if schema and schema != MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION:
                    raise MolrRuntimeBundleError(
                        f"Checkpoint schema mismatch in '{ckpt_path}': "
                        f"got '{schema}', expected '{MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION}'",
                    )
                ckpt_model = _npz_scalar_string(payload, "model_spec")
                if ckpt_model and ckpt_model != model_spec:
                    raise MolrRuntimeBundleError(
                        f"Checkpoint model mismatch in '{ckpt_path}': got '{ckpt_model}', expected '{model_spec}'",
                    )
        except MolrRuntimeBundleError:
            raise
        except Exception as exc:
            raise MolrRuntimeBundleError(f"Failed loading checkpoint '{ckpt_path}': {exc}") from exc

        out.append(
            {
                "layer": layer,
                "expert": expert,
                "path": str(ckpt_path),
            }
        )

    out.sort(key=lambda r: (int(r["layer"]), int(r["expert"])))
    return out


def load_runtime_bundle(
    *,
    bundle_dir: Path,
    runtime_config_path: Path,
    expected_model: str | None = None,
    require_explicit_enable: bool = True,
) -> dict[str, Any]:
    """Load and validate a packaged MoLR bundle for guarded runtime use.

    This helper is intended for Phase-4 opt-in runtime integration scaffolding.
    It validates compatibility contracts and resolves threshold selection from either
    an explicit threshold or a quality profile.
    """

    root = bundle_dir.expanduser().resolve()
    runtime_cfg_path = runtime_config_path.expanduser().resolve()
    if not root.is_dir():
        raise MolrRuntimeBundleError(f"Bundle directory does not exist: '{root}'")
    if not runtime_cfg_path.is_file():
        raise MolrRuntimeBundleError(f"Runtime config path is not a file: '{runtime_cfg_path}'")

    manifest_path = root / "molr_bundle_manifest.json"
    thresholds_path = root / "molr_thresholds.json"
    plan_path = root / "molr_plan.json"

    if not manifest_path.is_file():
        raise MolrRuntimeBundleError(f"Bundle manifest missing: '{manifest_path}'")
    if not thresholds_path.is_file():
        raise MolrRuntimeBundleError(f"Bundle thresholds missing: '{thresholds_path}'")
    if not plan_path.is_file():
        raise MolrRuntimeBundleError(f"Bundle plan missing: '{plan_path}'")

    manifest = _load_json(manifest_path)
    model_spec, _compatibility = _validate_manifest(manifest, expected_model)

    thresholds = _load_json(thresholds_path)
    thresholds_schema = str(thresholds.get("schema_version") or "")
    if thresholds_schema != MOLR_THRESHOLDS_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            "Bundle thresholds schema mismatch: "
            f"got '{thresholds_schema}', expected '{MOLR_THRESHOLDS_SCHEMA_VERSION}'",
        )
    thresholds_model = str(thresholds.get("model_spec") or "")
    if thresholds_model and thresholds_model != model_spec:
        raise MolrRuntimeBundleError(
            f"Bundle thresholds model mismatch: got '{thresholds_model}', expected '{model_spec}'",
        )

    plan = _load_json(plan_path)
    plan_schema = str(plan.get("schema_version") or "")
    if plan_schema != MOLR_PLAN_SCHEMA_VERSION:
        raise MolrRuntimeBundleError(
            f"Bundle plan schema mismatch: got '{plan_schema}', expected '{MOLR_PLAN_SCHEMA_VERSION}'",
        )
    plan_model = str(plan.get("model_spec") or "")
    if plan_model and plan_model != model_spec:
        raise MolrRuntimeBundleError(
            f"Bundle plan model mismatch: got '{plan_model}', expected '{model_spec}'",
        )

    runtime_cfg = _load_json(runtime_cfg_path)
    resolved_runtime_cfg = _validate_runtime_config(
        runtime_cfg,
        thresholds=thresholds,
        require_explicit_enable=require_explicit_enable,
    )

    checkpoints = _validate_checkpoint_rows(root, manifest, model_spec)

    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "thresholds_path": str(thresholds_path),
        "plan_path": str(plan_path),
        "runtime_config_path": str(runtime_cfg_path),
        "model_spec": model_spec,
        "runtime": resolved_runtime_cfg,
        "coverage": manifest.get("coverage", {}),
        "checkpoints": checkpoints,
    }
