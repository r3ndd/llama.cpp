#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.types import (
    MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION,
    MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
    MOLR_THRESHOLDS_SCHEMA_VERSION,
)


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrBundleError(RuntimeError):
    """Raised when Phase-3 bundle packaging fails."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrBundleError(f"Failed reading JSON '{path}': {exc}") from exc


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrBundleError(f"Failed writing JSON '{path}': {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _npz_scalar_string(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    try:
        if value.ndim == 0:
            return str(value.item())
    except Exception:
        pass
    return str(value)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-3 MoLR bundle packaging. Copies checkpoints + plan + thresholds into bundle/ "
            "and emits manifest with checksums and compatibility metadata."
        ),
    )
    parser.add_argument("--model", required=True, help="Model spec for compatibility metadata.")
    parser.add_argument("--plan-json", required=True, help="Path to molr_plan.json")
    parser.add_argument("--checkpoints", required=True, help="Path to checkpoints directory")
    parser.add_argument("--thresholds-json", required=True, help="Path to molr_thresholds.json")
    parser.add_argument("--out-dir", required=True, help="Output bundle directory")
    parser.add_argument(
        "--checkpoint-glob",
        default="molr_expert_*.npz",
        help="Checkpoint glob inside --checkpoints. Default: molr_expert_*.npz",
    )
    parser.add_argument(
        "--require-all-plan-experts",
        action="store_true",
        help="Fail if any plan expert is missing checkpoint coverage.",
    )
    args = parser.parse_args(argv)
    return args


def _validate_plan(path: Path, expected_model: str) -> dict[str, Any]:
    plan = _load_json(path)
    schema = str(plan.get("schema_version") or "")
    if schema != MOLR_PLAN_SCHEMA_VERSION:
        raise MolrBundleError(
            f"Plan schema mismatch: got '{schema}', expected '{MOLR_PLAN_SCHEMA_VERSION}'",
        )
    model_spec = str(plan.get("model_spec") or "")
    if model_spec and model_spec != expected_model:
        raise MolrBundleError(
            f"Plan model_spec mismatch: got '{model_spec}', expected '{expected_model}'",
        )
    return plan


def _validate_thresholds(path: Path, expected_model: str) -> dict[str, Any]:
    payload = _load_json(path)
    schema = str(payload.get("schema_version") or "")
    if schema != MOLR_THRESHOLDS_SCHEMA_VERSION:
        raise MolrBundleError(
            f"Threshold schema mismatch: got '{schema}', expected '{MOLR_THRESHOLDS_SCHEMA_VERSION}'",
        )
    model_spec = str(payload.get("model_spec") or "")
    if model_spec and model_spec != expected_model:
        raise MolrBundleError(
            f"Threshold model_spec mismatch: got '{model_spec}', expected '{expected_model}'",
        )
    return payload


def _checkpoint_key(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    if parts[0] != "molr" or parts[1] != "expert":
        return None
    try:
        layer = int(parts[2])
        expert = int(parts[3])
    except Exception:
        return None
    return layer, expert


def _plan_keys(plan: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for entry in plan.get("experts", []):
        out.add((int(entry.get("layer")), int(entry.get("expert"))))
    return out


def _validate_checkpoint(path: Path, expected_model: str) -> tuple[int, int]:
    try:
        import numpy as np

        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise MolrBundleError(f"Failed loading checkpoint NPZ '{path}': {exc}") from exc

    schema = _npz_scalar_string(payload, "schema_version")
    if schema is not None and schema != MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION:
        raise MolrBundleError(
            f"Checkpoint schema mismatch in '{path}': got '{schema}', expected '{MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION}'",
        )

    model_spec = _npz_scalar_string(payload, "model_spec")
    if model_spec is not None and model_spec != expected_model:
        raise MolrBundleError(
            f"Checkpoint model mismatch in '{path}': got '{model_spec}', expected '{expected_model}'",
        )

    if "layer" not in payload or "expert" not in payload:
        raise MolrBundleError(f"Checkpoint missing layer/expert arrays: '{path}'")
    layer = int(payload["layer"].item())
    expert = int(payload["expert"].item())
    return layer, expert


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)

        plan_path = Path(args.plan_json).expanduser().resolve()
        checkpoints_dir = Path(args.checkpoints).expanduser().resolve()
        thresholds_path = Path(args.thresholds_json).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve()

        if not plan_path.is_file():
            raise MolrBundleError(f"--plan-json path is not a file: '{plan_path}'")
        if not checkpoints_dir.is_dir():
            raise MolrBundleError(f"--checkpoints path is not a directory: '{checkpoints_dir}'")
        if not thresholds_path.is_file():
            raise MolrBundleError(f"--thresholds-json path is not a file: '{thresholds_path}'")

        plan = _validate_plan(plan_path, args.model)
        thresholds = _validate_thresholds(thresholds_path, args.model)

        checkpoint_paths = sorted(checkpoints_dir.glob(args.checkpoint_glob))
        if not checkpoint_paths:
            raise MolrBundleError(
                f"No checkpoints matched glob '{args.checkpoint_glob}' under '{checkpoints_dir}'",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_ckpt_dir = out_dir / "checkpoints"
        bundle_ckpt_dir.mkdir(parents=True, exist_ok=True)

        plan_out = out_dir / "molr_plan.json"
        thresholds_out = out_dir / "molr_thresholds.json"
        shutil.copy2(plan_path, plan_out)
        shutil.copy2(thresholds_path, thresholds_out)

        checkpoint_entries: list[dict[str, Any]] = []
        packaged_keys: set[tuple[int, int]] = set()

        for src in checkpoint_paths:
            key_from_name = _checkpoint_key(src)
            layer_npz, expert_npz = _validate_checkpoint(src, args.model)
            key_npz = (layer_npz, expert_npz)
            if key_from_name is not None and key_from_name != key_npz:
                raise MolrBundleError(
                    f"Checkpoint naming mismatch in '{src}': filename key={key_from_name}, npz key={key_npz}",
                )
            if key_npz in packaged_keys:
                raise MolrBundleError(f"Duplicate checkpoint for expert {key_npz}")
            packaged_keys.add(key_npz)

            dst = bundle_ckpt_dir / f"molr_expert_{layer_npz}_{expert_npz}.npz"
            shutil.copy2(src, dst)
            checkpoint_entries.append(
                {
                    "layer": layer_npz,
                    "expert": expert_npz,
                    "path": str(dst),
                    "sha256": _sha256_file(dst),
                }
            )

        checkpoint_entries.sort(key=lambda row: (int(row["layer"]), int(row["expert"])))
        plan_experts = _plan_keys(plan)
        missing_from_bundle = sorted(plan_experts - packaged_keys)

        if args.require_all_plan_experts and missing_from_bundle:
            preview = ", ".join(f"({l},{e})" for l, e in missing_from_bundle[:16])
            raise MolrBundleError(
                f"Missing checkpoints for {len(missing_from_bundle)} plan experts: {preview}",
            )

        manifest = {
            "schema_version": MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "bundle_layout_version": "molr_bundle.layout.v1",
            "model_spec": args.model,
            "compatibility": {
                "runtime_contract": "phase3_offline_bundle",
                "plan_schema_version": MOLR_PLAN_SCHEMA_VERSION,
                "thresholds_schema_version": MOLR_THRESHOLDS_SCHEMA_VERSION,
                "checkpoint_schema_version": MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
            },
            "inputs": {
                "plan_json": str(plan_path),
                "checkpoints_dir": str(checkpoints_dir),
                "thresholds_json": str(thresholds_path),
            },
            "artifacts": {
                "plan_json": {
                    "path": str(plan_out),
                    "sha256": _sha256_file(plan_out),
                },
                "thresholds_json": {
                    "path": str(thresholds_out),
                    "sha256": _sha256_file(thresholds_out),
                },
                "checkpoints": checkpoint_entries,
            },
            "coverage": {
                "plan_experts_total": len(plan_experts),
                "checkpoint_experts_total": len(packaged_keys),
                "missing_plan_experts_total": len(missing_from_bundle),
                "missing_plan_experts": [
                    {"layer": int(layer), "expert": int(expert)} for layer, expert in missing_from_bundle
                ],
            },
            "threshold_summary": {
                "quality_profiles": thresholds.get("quality_profiles", {}),
                "lookup_table_entries": len(thresholds.get("lookup_table", [])),
            },
        }
        manifest_path = out_dir / "molr_bundle_manifest.json"
        _save_json(manifest_path, manifest)

        print(
            "[molr-package] wrote "
            f"checkpoints={len(packaged_keys)} missing={len(missing_from_bundle)} "
            f"-> {manifest_path}",
        )
        return EXIT_OK

    except MolrBundleError as exc:
        print(f"[error:molr-package] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
