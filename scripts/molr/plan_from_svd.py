#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_svd.svd_metrics import SPECTRAL_ENERGY_RANK_FRACTIONS, rank_from_fraction

from molr.types import MOLR_PLAN_SCHEMA_VERSION, SVD_REPORT_SCHEMA_VERSION


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


class MolrPlanError(RuntimeError):
    """Raised when MoLR planning from SVD report fails."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrPlanError(f"Failed reading JSON '{path}': {exc}") from exc


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrPlanError(f"Failed writing JSON '{path}': {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Phase-1 molr_plan.json from an existing svd_report.json "
            "using nearest-above explained-energy rank selection and strided init partitions."
        ),
    )
    parser.add_argument("--svd-json", required=True, help="Path to svd_report.json.")
    parser.add_argument(
        "--target-energy",
        type=float,
        default=0.90,
        help="Target explained spectral energy in (0, 1]. Default: 0.90.",
    )
    parser.add_argument(
        "--k-components",
        type=int,
        default=4,
        help="Number of MoLR components K used for strided init map. Default: 4.",
    )
    parser.add_argument("--out-json", required=True, help="Output path for molr_plan.json.")
    args = parser.parse_args(argv)

    if args.target_energy <= 0.0 or args.target_energy > 1.0:
        parser.error("--target-energy must be in (0, 1].")
    if args.k_components <= 0:
        parser.error("--k-components must be > 0.")
    return args


def _validate_svd_report_schema(svd_report: dict[str, Any]) -> None:
    schema_version = str(svd_report.get("schema_version") or "")
    if schema_version != SVD_REPORT_SCHEMA_VERSION:
        raise MolrPlanError(
            f"Unexpected svd_report schema_version='{schema_version}', expected '{SVD_REPORT_SCHEMA_VERSION}'.",
        )

    run = svd_report.get("run", {})
    fidelity_mode = str(run.get("fidelity_mode") or "")
    if fidelity_mode != "full_svd":
        raise MolrPlanError(
            f"svd_report run.fidelity_mode must be 'full_svd', got '{fidelity_mode}'.",
        )

    per_matrix = svd_report.get("per_matrix", [])
    if not isinstance(per_matrix, list) or not per_matrix:
        raise MolrPlanError("svd_report.per_matrix is empty; cannot build MoLR plan.")


def _matrix_shape(record: dict[str, Any]) -> tuple[int, int]:
    shape_raw = record.get("shape")
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 2:
        raise MolrPlanError(f"Invalid matrix shape in tensor '{record.get('tensor')}': {shape_raw}")
    try:
        m = int(shape_raw[0])
        n = int(shape_raw[1])
    except Exception as exc:
        raise MolrPlanError(f"Non-integer matrix shape in tensor '{record.get('tensor')}': {shape_raw}") from exc
    if m <= 0 or n <= 0:
        raise MolrPlanError(f"Non-positive matrix shape in tensor '{record.get('tensor')}': {shape_raw}")
    return m, n


def _select_rank(
    *,
    m: int,
    n: int,
    target_energy: float,
    explained_curve: list[float],
) -> tuple[int, float, float, bool, bool]:
    """Return (rank, selected_fraction, selected_energy, target_met, clamped_to_full_rank)."""
    if len(explained_curve) != len(SPECTRAL_ENERGY_RANK_FRACTIONS):
        raise MolrPlanError(
            "Unexpected explained_spectral_energy_rank_fractions length: "
            f"got {len(explained_curve)}, expected {len(SPECTRAL_ENERGY_RANK_FRACTIONS)}."
        )

    selected_fraction = float(SPECTRAL_ENERGY_RANK_FRACTIONS[-1])
    selected_energy = float(explained_curve[-1])
    target_met = selected_energy >= target_energy

    for frac, energy in zip(SPECTRAL_ENERGY_RANK_FRACTIONS, explained_curve):
        e = float(energy)
        if e >= target_energy:
            selected_fraction = float(frac)
            selected_energy = e
            target_met = True
            break

    rank = int(rank_from_fraction(m, n, selected_fraction))
    min_dim = min(m, n)
    clamped_to_full_rank = False
    if not target_met:
        rank = min_dim
        selected_fraction = 1.0
        selected_energy = 1.0
        clamped_to_full_rank = True

    rank = max(1, min(rank, min_dim))
    return rank, selected_fraction, selected_energy, target_met, clamped_to_full_rank


def _strided_partition(rank: int, k_components: int) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    component_rank_counts: list[int] = []
    for comp in range(k_components):
        rank_indices = list(range(comp, rank, k_components))
        assignments.append({"component": comp, "rank_indices": rank_indices})
        component_rank_counts.append(len(rank_indices))

    return {
        "strategy": "strided",
        "component_assignments": assignments,
        "component_rank_counts": component_rank_counts,
        "frobenius_equalization_required": True,
    }


def _extract_expert_key(record: dict[str, Any]) -> tuple[int, int]:
    layer = record.get("layer")
    expert = record.get("expert")
    if layer is None or expert is None:
        raise MolrPlanError(
            f"Missing layer/expert metadata for tensor '{record.get('tensor')}', cannot plan per-expert ranks."
        )
    return int(layer), int(expert)


def _build_plan(svd_report: dict[str, Any], *, target_energy: float, k_components: int, svd_json_path: Path) -> dict[str, Any]:
    per_matrix = svd_report.get("per_matrix", [])
    run = svd_report.get("run", {})

    experts_map: dict[tuple[int, int], list[dict[str, Any]]] = {}
    failures_unreached_target = 0
    skipped: list[dict[str, Any]] = []

    for idx, record in enumerate(per_matrix):
        try:
            layer, expert = _extract_expert_key(record)
            m, n = _matrix_shape(record)
            curve = [float(x) for x in record.get("explained_spectral_energy_rank_fractions", [])]
            rank, selected_frac, selected_energy, target_met, clamped = _select_rank(
                m=m,
                n=n,
                target_energy=target_energy,
                explained_curve=curve,
            )
            if not target_met:
                failures_unreached_target += 1

            matrix_entry = {
                "role": str(record.get("role") or "unknown"),
                "tensor": str(record.get("tensor") or ""),
                "source_tensor": str(record.get("source_tensor") or ""),
                "shape": [m, n],
                "min_dimension": min(m, n),
                "rank": rank,
                "k_components": k_components,
                "fro_norm": float(record.get("fro_norm") or 0.0),
                "singular_value_count": int(record.get("singular_value_count") or 0),
                "target_energy": target_energy,
                "target_energy_met_on_fraction_grid": target_met,
                "selected_rank_fraction": selected_frac,
                "selected_explained_energy": selected_energy,
                "energy_curve_ref": {
                    "source": "svd_report.per_matrix.explained_spectral_energy_rank_fractions",
                    "per_matrix_index": idx,
                },
                "init_partition": _strided_partition(rank=rank, k_components=k_components),
                "selection_notes": {
                    "policy": "nearest_above_fraction_grid",
                    "clamped_to_full_rank": clamped,
                },
            }
            experts_map.setdefault((layer, expert), []).append(matrix_entry)
        except MolrPlanError as exc:
            skipped.append(
                {
                    "per_matrix_index": idx,
                    "tensor": str(record.get("tensor") or ""),
                    "reason": str(exc),
                }
            )

    experts: list[dict[str, Any]] = []
    for (layer, expert), matrices in sorted(experts_map.items(), key=lambda kv: kv[0]):
        matrices.sort(key=lambda m: (m["role"], m["tensor"]))
        experts.append(
            {
                "layer": layer,
                "expert": expert,
                "matrices": matrices,
            }
        )

    return {
        "schema_version": MOLR_PLAN_SCHEMA_VERSION,
        "created_at_utc": _now_utc_iso(),
        "source_svd": {
            "path": str(svd_json_path),
            "schema_version": str(svd_report.get("schema_version") or ""),
            "model_spec": run.get("model_spec"),
            "fidelity_mode": run.get("fidelity_mode"),
            "git_commit": run.get("git_commit"),
        },
        "model_spec": run.get("model_spec"),
        "target_energy": target_energy,
        "default_k": k_components,
        "rank_selection_policy": {
            "name": "nearest_above_fraction_grid",
            "fraction_grid": list(SPECTRAL_ENERGY_RANK_FRACTIONS),
            "fallback_when_unreached": "full_rank",
        },
        "experts": experts,
        "summary": {
            "experts_total": len(experts),
            "matrices_total": sum(len(e["matrices"]) for e in experts),
            "matrices_unreached_target_on_fraction_grid": failures_unreached_target,
            "per_matrix_records_skipped": len(skipped),
        },
        "skipped_per_matrix_records": skipped,
    }


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        svd_json_path = Path(args.svd_json).expanduser().resolve()
        out_json_path = Path(args.out_json).expanduser().resolve()
        if not svd_json_path.is_file():
            raise MolrPlanError(f"--svd-json path is not a file: '{svd_json_path}'")

        svd_report = _load_json(svd_json_path)
        _validate_svd_report_schema(svd_report)
        plan = _build_plan(
            svd_report,
            target_energy=float(args.target_energy),
            k_components=int(args.k_components),
            svd_json_path=svd_json_path,
        )
        _save_json(out_json_path, plan)

        print(
            "[molr-plan] wrote "
            f"experts={plan['summary']['experts_total']} "
            f"matrices={plan['summary']['matrices_total']} "
            f"target_energy={plan['target_energy']:.4f} -> {out_json_path}",
        )
        return EXIT_OK
    except MolrPlanError as exc:
        print(f"[error:molr-plan] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
