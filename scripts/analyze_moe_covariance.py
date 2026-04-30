#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from numbers import Integral
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from moe_svd.gguf_discovery import DiscoveryError, discover_expert_matrices
from moe_svd.model_resolver import ModelResolutionError, resolve_model_path


def _ensure_gguf_import() -> Any:
    try:
        import gguf  # type: ignore

        return gguf
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        gguf_py = repo_root / "gguf-py"
        if gguf_py.is_dir() and str(gguf_py) not in sys.path:
            sys.path.insert(0, str(gguf_py))

    import gguf  # type: ignore  # noqa: E402

    return gguf


def _get_field_contents(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    try:
        return field.contents()
    except Exception:
        return None


def _parse_index_spec(spec: str, *, field_name: str) -> list[int]:
    values: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid {field_name} spec '{spec}': empty element")
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo = int(lo_str)
            hi = int(hi_str)
            if lo > hi:
                raise ValueError(f"invalid {field_name} range '{part}': start > end")
            for v in range(lo, hi + 1):
                values.add(v)
        else:
            values.add(int(part))
    return sorted(values)


def _parse_targets_spec(spec: str, observed_roles: set[str]) -> list[str]:
    default_roles = {"gate", "up", "down"}
    text = (spec or "").strip().lower()
    if not text or text == "all":
        return sorted(default_roles | observed_roles)

    out: list[str] = []
    seen: set[str] = set()
    for raw_part in text.split(","):
        role = raw_part.strip()
        if not role:
            raise ValueError(f"invalid targets spec '{spec}': empty element")
        if role == "all":
            raise ValueError(f"invalid targets spec '{spec}': cannot combine 'all' with other roles")
        if role not in seen:
            out.append(role)
            seen.add(role)
    return out


def _logical_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in reversed(tensor.shape.tolist()))


def _tensor_to_array_f64(tensor: Any) -> np.ndarray:
    shape = _logical_shape(tensor)
    arr = np.asarray(tensor.data)
    return arr.astype(np.float64, copy=False).reshape(shape)


def _participation_coefficient_from_singular_values(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    s = singular_values.astype(np.float64, copy=False)
    denom = float(np.sum(np.square(s, dtype=np.float64), dtype=np.float64))
    if denom == 0.0:
        return 0.0
    numer = float(np.square(np.sum(s, dtype=np.float64), dtype=np.float64))
    return numer / denom


@dataclass(slots=True)
class CovTarget:
    tid: str
    layer: int
    expert: int
    role: str
    tensor_name: str
    role_variant: str | None
    dim: int
    n_vectors: int
    has_cov_tensor: bool
    participation_coefficient: float
    computed_covariance: bool


def _describe_number(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.6g}"
    return f"min={min(values):.6g} mean={statistics.fmean(values):.6g} max={max(values):.6g}"


def _print_list(label: str, values: list[int] | list[str]) -> None:
    if values:
        print(f"{label}: {', '.join(str(v) for v in values)}")
    else:
        print(f"{label}: (none)")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a llama-imatrix MoE covariance GGUF file and report per-target coverage, "
            "per-matrix dimensions/sample counts, participation coefficients, and grouped summaries."
        ),
    )
    parser.add_argument("cov_gguf", help="Path to covariance GGUF file (general.type=moe_covariance).")
    parser.add_argument(
        "model",
        help=(
            "Model path or model spec used to compute covariance. "
            "Model spec format: <repo_id>:<filename_or_quant>."
        ),
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=50,
        help="Maximum missing layer/expert/role combinations to print (default: 50).",
    )
    return parser


def _resolve_model_gguf_path(model_arg: str) -> Path:
    maybe_path = Path(model_arg).expanduser()
    if maybe_path.is_file():
        return maybe_path.resolve()

    try:
        resolved = resolve_model_path(model_arg, cache_dir=None)
    except ModelResolutionError as exc:
        raise RuntimeError(
            "failed to resolve model argument as path or model spec "
            f"'{model_arg}': {exc}",
        ) from exc

    return Path(resolved.local_path).resolve()


def _canonical_role(role: str) -> str:
    role_l = role.lower()
    if role_l == "w1":
        return "gate"
    if role_l == "w2":
        return "down"
    if role_l == "w3":
        return "up"
    return role_l


def _build_expected_universe(
    *,
    model_path: Path,
    observed_roles: set[str],
) -> tuple[list[int], list[int], list[str], dict[str, Any]]:
    try:
        discovery = discover_expert_matrices(
            gguf_path=str(model_path),
            include=[],
            exclude=[],
        )
    except DiscoveryError as exc:
        raise RuntimeError(f"failed to load model architecture from '{model_path}': {exc}") from exc

    metadata = discovery.metadata
    block_count = metadata.get("block_count")
    expert_count = metadata.get("expert_count")

    if isinstance(block_count, Integral) and int(block_count) > 0:
        layers_expected = list(range(int(block_count)))
    else:
        layers_expected = sorted({int(ref.layer) for ref in discovery.candidates if ref.layer is not None})

    if isinstance(expert_count, Integral) and int(expert_count) > 0:
        experts_expected = list(range(int(expert_count)))
    else:
        experts_expected = sorted({int(ref.expert) for ref in discovery.candidates if ref.expert is not None})

    roles_expected = sorted({_canonical_role(ref.role) for ref in discovery.candidates if isinstance(ref.role, str)})
    if not roles_expected:
        roles_expected = sorted({_canonical_role(role) for role in observed_roles})

    return layers_expected, experts_expected, roles_expected, metadata


def _collect_targets(reader: Any) -> list[CovTarget]:
    tensor_lookup = {t.name: t for t in reader.tensors}
    target_count = _get_field_contents(reader, "moe_cov.target_count")
    if not isinstance(target_count, int):
        raise RuntimeError("missing or invalid metadata key: moe_cov.target_count")

    out: list[CovTarget] = []
    for idx in range(target_count):
        tid = f"t{idx}"
        prefix = f"moe_cov.target.{tid}"
        layer = _get_field_contents(reader, f"{prefix}.layer")
        expert = _get_field_contents(reader, f"{prefix}.expert")
        role = _get_field_contents(reader, f"{prefix}.role")
        dim = _get_field_contents(reader, f"{prefix}.dim")
        tensor_name = _get_field_contents(reader, f"{prefix}.tensor_name")
        role_variant = _get_field_contents(reader, f"{prefix}.role_variant")

        if not isinstance(layer, int) or not isinstance(expert, int) or not isinstance(dim, int):
            raise RuntimeError(f"invalid target metadata for {tid}: layer/expert/dim")
        if not isinstance(role, str) or not isinstance(tensor_name, str):
            raise RuntimeError(f"invalid target metadata for {tid}: role/tensor_name")

        n_tensor = tensor_lookup.get(f"moe_cov.{tid}.n")
        cov_tensor = tensor_lookup.get(f"moe_cov.{tid}.cov_pop")
        if n_tensor is None:
            raise RuntimeError(f"missing tensor: moe_cov.{tid}.n")

        n_arr = np.asarray(n_tensor.data)
        if n_arr.size != 1:
            raise RuntimeError(f"invalid n tensor shape for {tid}")
        n_vectors = int(n_arr.reshape(1)[0])

        participation = float("nan")
        has_cov_tensor = cov_tensor is not None
        if cov_tensor is not None:
            cov = _tensor_to_array_f64(cov_tensor)
            if cov.ndim != 2:
                raise RuntimeError(f"covariance tensor for {tid} is not 2D")
            if cov.shape[0] != dim or cov.shape[1] != dim:
                raise RuntimeError(
                    f"covariance tensor shape mismatch for {tid}: expected ({dim}, {dim}), got {cov.shape}",
                )
            singular_values = np.linalg.svd(cov, full_matrices=False, compute_uv=False)
            participation = _participation_coefficient_from_singular_values(singular_values)

        if n_vectors < 0:
            raise RuntimeError(f"invalid vector count for {tid}: n must be >= 0")

        out.append(
            CovTarget(
                tid=tid,
                layer=layer,
                expert=expert,
                role=role,
                tensor_name=tensor_name,
                role_variant=role_variant if isinstance(role_variant, str) else None,
                dim=dim,
                n_vectors=n_vectors,
                has_cov_tensor=has_cov_tensor,
                participation_coefficient=participation,
                computed_covariance=n_vectors >= 2 and has_cov_tensor,
            ),
        )

    out.sort(key=lambda t: (t.layer, t.expert, t.role, t.tid))
    return out


def main(argv: list[str]) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.max_missing < 0:
        parser.error("--max-missing must be >= 0")

    gguf = _ensure_gguf_import()

    cov_path = Path(args.cov_gguf).expanduser().resolve()
    if not cov_path.is_file():
        print(f"error: file not found: {cov_path}", file=sys.stderr)
        return 2

    try:
        reader = gguf.GGUFReader(str(cov_path), mode="r")
    except Exception as exc:
        print(f"error: failed to read GGUF file '{cov_path}': {exc}", file=sys.stderr)
        return 2

    gtype = _get_field_contents(reader, "general.type")
    if gtype != "moe_covariance":
        print(
            f"error: GGUF general.type must be 'moe_covariance', got: {gtype!r}",
            file=sys.stderr,
        )
        return 2

    try:
        model_path = _resolve_model_gguf_path(args.model)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        targets = _collect_targets(reader)
    except Exception as exc:
        print(f"error: failed to parse covariance targets: {exc}", file=sys.stderr)
        return 2

    if not targets:
        print("No covariance targets found (target_count=0).")
        return 0

    precision = _get_field_contents(reader, "moe_cov.precision")
    convention = _get_field_contents(reader, "moe_cov.convention")
    fingerprint = _get_field_contents(reader, "moe_cov.model_fingerprint")
    created_at = _get_field_contents(reader, "moe_cov.created_at")
    updated_at = _get_field_contents(reader, "moe_cov.updated_at")

    roles_obs = {t.role for t in targets}

    try:
        layers_expected, experts_expected, roles_expected, model_metadata = _build_expected_universe(
            model_path=model_path,
            observed_roles=roles_obs,
        )
    except Exception as exc:
        print(f"error: failed to determine expected model universe: {exc}", file=sys.stderr)
        return 2

    if not layers_expected:
        layers_expected = sorted({t.layer for t in targets})
    if not experts_expected:
        experts_expected = sorted({t.expert for t in targets})
    if not roles_expected:
        roles_expected = sorted({_canonical_role(t.role) for t in targets})

    for t in targets:
        t.role = _canonical_role(t.role)

    roles_obs = {t.role for t in targets}

    filter_layers = _get_field_contents(reader, "moe_cov.filters.layers")
    filter_experts = _get_field_contents(reader, "moe_cov.filters.experts")
    filter_targets = _get_field_contents(reader, "moe_cov.filters.targets")

    computed_targets = [t for t in targets if t.computed_covariance]

    computed_layers = sorted({t.layer for t in computed_targets})
    computed_experts = sorted({t.expert for t in computed_targets})
    computed_roles = sorted({t.role for t in computed_targets})

    not_computed_layers = [v for v in layers_expected if v not in set(computed_layers)]
    not_computed_experts = [v for v in experts_expected if v not in set(computed_experts)]
    not_computed_roles = [v for v in roles_expected if v not in set(computed_roles)]

    by_triplet: dict[tuple[int, int, str], list[CovTarget]] = {}
    for t in targets:
        by_triplet.setdefault((t.layer, t.expert, t.role), []).append(t)

    missing_triplets: list[tuple[int, int, str]] = []
    triplets_with_computed = 0
    for layer in layers_expected:
        for expert in experts_expected:
            for role in roles_expected:
                group = by_triplet.get((layer, expert, role), [])
                if any(x.computed_covariance for x in group):
                    triplets_with_computed += 1
                else:
                    missing_triplets.append((layer, expert, role))

    print("=== MoE Covariance GGUF Report ===")
    print(f"File: {cov_path}")
    print(f"Model: {model_path}")
    print(f"Type: {gtype}  precision={precision}  convention={convention}")
    print(
        "Model architecture: "
        f"{model_metadata.get('architecture')} "
        f"block_count={model_metadata.get('block_count')} "
        f"expert_count={model_metadata.get('expert_count')} "
        f"expert_used_count={model_metadata.get('expert_used_count')}",
    )
    print(f"Model fingerprint: {fingerprint}")
    if isinstance(created_at, str):
        print(f"Created at: {created_at}")
    if isinstance(updated_at, str):
        print(f"Updated at: {updated_at}")
    print()

    print("-- Coverage summary --")
    if isinstance(filter_layers, str) or isinstance(filter_experts, str) or isinstance(filter_targets, str):
        print(
            "Covariance filter metadata: "
            f"layers={filter_layers!r} experts={filter_experts!r} targets={filter_targets!r}",
        )
    _print_list("Expected layers", layers_expected)
    _print_list("Expected experts", experts_expected)
    _print_list("Expected targets", roles_expected)
    _print_list("Layers with computed covariance", computed_layers)
    _print_list("Layers without computed covariance", not_computed_layers)
    _print_list("Experts with computed covariance", computed_experts)
    _print_list("Experts without computed covariance", not_computed_experts)
    _print_list("Targets with computed covariance", computed_roles)
    _print_list("Targets without computed covariance", not_computed_roles)
    total_expected_triplets = len(layers_expected) * len(experts_expected) * len(roles_expected)
    print(f"Expected layer/expert/target combinations: {total_expected_triplets}")
    print(f"Combinations with computed covariance: {triplets_with_computed}")
    print(f"Combinations without computed covariance: {len(missing_triplets)}")
    if missing_triplets and args.max_missing > 0:
        print(f"Missing combinations (first {min(args.max_missing, len(missing_triplets))}):")
        for layer, expert, role in missing_triplets[: args.max_missing]:
            print(f"  - layer={layer} expert={expert} target={role}")
    print()

    print("-- Per covariance matrix --")
    for t in targets:
        status = "computed" if t.computed_covariance else "not-computed"
        variant = f" variant={t.role_variant}" if t.role_variant else ""
        pc = "nan" if not math.isfinite(t.participation_coefficient) else f"{t.participation_coefficient:.6g}"
        print(
            f"L{t.layer:03d} E{t.expert:03d} {t.role:<5} n={t.n_vectors:<8d} dim={t.dim:<6d} "
            f"pc={pc:<10} status={status} tid={t.tid}{variant} tensor={t.tensor_name}",
        )
    print()

    print("-- Group-level statistics --")
    print(f"Target records: {len(targets)}")
    print(f"Computed targets (n >= 2): {len(computed_targets)}")
    print(f"Not-computed targets (n < 2): {len(targets) - len(computed_targets)}")

    all_ns = [float(t.n_vectors) for t in targets]
    all_dims = [float(t.dim) for t in targets]
    all_pcs = [t.participation_coefficient for t in targets if math.isfinite(t.participation_coefficient)]
    computed_pcs = [t.participation_coefficient for t in computed_targets if math.isfinite(t.participation_coefficient)]

    print(f"Vectors per target: {_describe_number(all_ns)}")
    print(f"Dimensionality per target: {_describe_number(all_dims)}")
    print(f"Participation coefficient (all targets): {_describe_number(all_pcs)}")
    print(f"Participation coefficient (computed targets): {_describe_number(computed_pcs)}")

    roles_for_group = sorted({t.role for t in targets})
    for role in roles_for_group:
        subset = [t for t in targets if t.role == role]
        subset_comp = [t for t in subset if t.computed_covariance]
        subset_ns = [float(t.n_vectors) for t in subset]
        subset_dims = [float(t.dim) for t in subset]
        subset_pcs = [t.participation_coefficient for t in subset if math.isfinite(t.participation_coefficient)]
        print(
            f"  role={role}: total={len(subset)} computed={len(subset_comp)} "
            f"n[{_describe_number(subset_ns)}] dim[{_describe_number(subset_dims)}] pc[{_describe_number(subset_pcs)}]",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
