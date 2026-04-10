#!/usr/bin/env python3

"""Build Algorithm-2 MoE lookup tables from MoE trace NPZ v1 artifacts.

This tool consumes one or more llama.cpp MoE trace NPZ files and emits:
1) a per-layer shared centroid/residual lookup sidecar NPZ, and
2) a replaced-expert set JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TraceData:
    path: Path
    layer_ids: np.ndarray
    token_ids: np.ndarray
    h_pre_moe: np.ndarray
    topk_ids: np.ndarray
    topk_weights: np.ndarray
    y_full: np.ndarray
    y_kept: Optional[np.ndarray]
    residual_target: Optional[np.ndarray]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Algorithm-2 shared MoE lookup tables from trace NPZ v1 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Input MoE trace NPZ v1 path (repeat for multiple files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output lookup sidecar NPZ path.",
    )
    parser.add_argument(
        "--output-replaced-experts",
        type=Path,
        default=None,
        help="Output JSON path for per-layer replaced expert IDs.",
    )
    parser.add_argument(
        "--replaced-experts-json",
        type=Path,
        default=None,
        help="Optional input JSON describing per-layer replaced experts.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma-separated layer list/ranges (e.g. 0,1,3-5). Empty=all traced layers.",
    )
    parser.add_argument(
        "--clusters-per-layer",
        type=int,
        default=1024,
        help="Target number of shared centroids per layer.",
    )
    parser.add_argument(
        "--kmeans-iters",
        type=int,
        default=25,
        help="Maximum k-means refinement iterations.",
    )
    parser.add_argument(
        "--kmeans-max-samples-per-layer",
        type=int,
        default=250000,
        help="Max layer rows sampled for centroid training (<=0 means all rows).",
    )
    parser.add_argument(
        "--distance-batch-size",
        type=int,
        default=4096,
        help="Batch size for nearest-centroid distance computation.",
    )
    parser.add_argument(
        "--replace-ratio",
        type=float,
        default=0.10,
        help="Fraction of experts replaced per layer when no JSON set is provided.",
    )
    parser.add_argument(
        "--replace-heuristic",
        choices=["least-routed"],
        default="least-routed",
        help="Heuristic for selecting replaced experts when not provided explicitly.",
    )
    parser.add_argument(
        "--residual-source",
        choices=["auto", "residual_target", "y_full_minus_y_kept", "proxy_replaced_mass_y_full"],
        default="auto",
        help=(
            "Residual target source: `residual_target` array, `y_full - y_kept`, "
            "or proxy `router_mass_replaced * y_full` when optional arrays are unavailable."
        ),
    )
    parser.add_argument(
        "--scaling-mode",
        choices=["none", "router_mass_replaced"],
        default="router_mass_replaced",
        help="Runtime scaling assumption used when training residual table values.",
    )
    parser.add_argument(
        "--min-replaced-mass",
        type=float,
        default=1e-6,
        help="Rows with replaced mass <= threshold are ignored for mass-normalized residual aggregation.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    return parser.parse_args()


def _read_metadata_json(path: Path) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "metadata.json" not in zf.namelist():
                return {}
            raw = zf.read("metadata.json")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid NPZ/ZIP file") from exc

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        print(f"warning: '{path}': metadata.json is not valid UTF-8 JSON; ignoring metadata", file=sys.stderr)
        return {}

    if not isinstance(parsed, dict):
        raise ValueError(f"'{path}': metadata.json must be a JSON object")
    return parsed


def _require_array(npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in npz:
        raise ValueError(f"missing required array '{name}'")
    return npz[name]


def load_trace_npz(path: Path) -> TraceData:
    if not path.exists():
        raise ValueError(f"input NPZ does not exist: {path}")

    metadata = _read_metadata_json(path)
    if metadata:
        fmt = metadata.get("format_version")
        if fmt is not None and int(fmt) != 1:
            raise ValueError(f"{path}: unsupported trace metadata format_version={fmt}, expected 1")

    try:
        with np.load(path, allow_pickle=False) as npz:
            layer_ids = _require_array(npz, "layer_ids")
            token_ids = _require_array(npz, "token_ids")
            h_pre_moe = _require_array(npz, "h_pre_moe")
            topk_ids = _require_array(npz, "topk_ids")
            topk_weights = _require_array(npz, "topk_weights")
            y_full = _require_array(npz, "y_full")
            y_kept = npz["y_kept"] if "y_kept" in npz else None
            residual_target = npz["residual_target"] if "residual_target" in npz else None
    except Exception as exc:
        raise ValueError(f"failed to load trace NPZ '{path}': {exc}") from exc

    layer_ids = np.asarray(layer_ids, dtype=np.int32)
    token_ids = np.asarray(token_ids, dtype=np.int32)
    h_pre_moe = np.asarray(h_pre_moe, dtype=np.float32)
    topk_ids = np.asarray(topk_ids, dtype=np.int32)
    topk_weights = np.asarray(topk_weights, dtype=np.float32)
    y_full = np.asarray(y_full, dtype=np.float32)
    y_kept = None if y_kept is None else np.asarray(y_kept, dtype=np.float32)
    residual_target = None if residual_target is None else np.asarray(residual_target, dtype=np.float32)

    validate_trace_shapes(path, layer_ids, token_ids, h_pre_moe, topk_ids, topk_weights, y_full, y_kept, residual_target)

    return TraceData(
        path=path,
        layer_ids=layer_ids,
        token_ids=token_ids,
        h_pre_moe=h_pre_moe,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        y_full=y_full,
        y_kept=y_kept,
        residual_target=residual_target,
        metadata=metadata,
    )


def validate_trace_shapes(
    path: Path,
    layer_ids: np.ndarray,
    token_ids: np.ndarray,
    h_pre_moe: np.ndarray,
    topk_ids: np.ndarray,
    topk_weights: np.ndarray,
    y_full: np.ndarray,
    y_kept: Optional[np.ndarray],
    residual_target: Optional[np.ndarray],
) -> None:
    if layer_ids.ndim != 1:
        raise ValueError(f"{path}: layer_ids must be 1D")
    if token_ids.ndim != 1:
        raise ValueError(f"{path}: token_ids must be 1D")

    n_rows = int(layer_ids.shape[0])
    if token_ids.shape[0] != n_rows:
        raise ValueError(f"{path}: token_ids length mismatch vs layer_ids")

    if h_pre_moe.ndim != 2:
        raise ValueError(f"{path}: h_pre_moe must have shape [N, n_embd]")
    if y_full.ndim != 2:
        raise ValueError(f"{path}: y_full must have shape [N, n_embd]")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError(f"{path}: topk_ids/topk_weights must have shape [N, k]")

    if h_pre_moe.shape[0] != n_rows:
        raise ValueError(f"{path}: h_pre_moe row count mismatch")
    if y_full.shape[0] != n_rows:
        raise ValueError(f"{path}: y_full row count mismatch")
    if topk_ids.shape[0] != n_rows or topk_weights.shape[0] != n_rows:
        raise ValueError(f"{path}: topk_ids/topk_weights row count mismatch")
    if topk_ids.shape[1] != topk_weights.shape[1]:
        raise ValueError(f"{path}: topk_ids and topk_weights k dimension mismatch")
    if y_full.shape[1] != h_pre_moe.shape[1]:
        raise ValueError(f"{path}: y_full and h_pre_moe n_embd mismatch")

    if y_kept is not None:
        if y_kept.shape != y_full.shape:
            raise ValueError(f"{path}: y_kept shape must match y_full")
    if residual_target is not None:
        if residual_target.shape != y_full.shape:
            raise ValueError(f"{path}: residual_target shape must match y_full")

    if np.any(topk_ids < 0):
        raise ValueError(f"{path}: topk_ids contains negative expert IDs")
    if not np.all(np.isfinite(topk_weights)):
        raise ValueError(f"{path}: topk_weights contains non-finite values")


def parse_layers_arg(layers_arg: str, available_layers: np.ndarray) -> List[int]:
    available = sorted(int(x) for x in np.unique(available_layers))
    if not layers_arg.strip():
        return available

    selected: set[int] = set()
    for part in layers_arg.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            lo_s, hi_s = p.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid layer range '{p}'")
            for x in range(lo, hi + 1):
                selected.add(x)
        else:
            selected.add(int(p))

    invalid = sorted(x for x in selected if x not in set(available))
    if invalid:
        raise ValueError(f"requested layers not present in traces: {invalid}")
    return sorted(selected)


def infer_dim_consistency(traces: List[TraceData]) -> Tuple[int, int, int]:
    n_embd = int(traces[0].h_pre_moe.shape[1])
    n_topk = int(traces[0].topk_ids.shape[1])
    observed_n_expert = int(np.max(traces[0].topk_ids)) + 1
    metadata_n_expert: Optional[int] = None

    for t in traces[1:]:
        if t.h_pre_moe.shape[1] != n_embd:
            raise ValueError(f"n_embd mismatch in '{t.path}'")
        if t.topk_ids.shape[1] != n_topk:
            raise ValueError(f"top-k width mismatch in '{t.path}'")
        observed_n_expert = max(observed_n_expert, int(np.max(t.topk_ids)) + 1)

    for t in traces:
        meta_n_embd = t.metadata.get("n_embd")
        if meta_n_embd is not None and int(meta_n_embd) != n_embd:
            raise ValueError(f"{t.path}: metadata n_embd mismatch ({meta_n_embd} vs {n_embd})")

        meta_n_expert = t.metadata.get("n_expert")
        if meta_n_expert is not None:
            meta_n_expert_i = int(meta_n_expert)
            if meta_n_expert_i < observed_n_expert:
                raise ValueError(
                    f"{t.path}: metadata n_expert ({meta_n_expert_i}) smaller than traced expert IDs ({observed_n_expert})"
                )
            if metadata_n_expert is None:
                metadata_n_expert = meta_n_expert_i
            elif metadata_n_expert != meta_n_expert_i:
                raise ValueError(
                    f"{t.path}: metadata n_expert mismatch ({meta_n_expert_i} vs {metadata_n_expert})"
                )

        meta_n_expert_used = t.metadata.get("n_expert_used")
        if meta_n_expert_used is not None and int(meta_n_expert_used) != n_topk:
            raise ValueError(f"{t.path}: metadata n_expert_used mismatch ({meta_n_expert_used} vs top-k width {n_topk})")

    n_expert = metadata_n_expert if metadata_n_expert is not None else observed_n_expert
    return n_embd, n_topk, n_expert


def validate_args(args: argparse.Namespace) -> None:
    if args.clusters_per_layer <= 0:
        raise ValueError("--clusters-per-layer must be > 0")
    if args.kmeans_iters <= 0:
        raise ValueError("--kmeans-iters must be > 0")
    if args.distance_batch_size == 0:
        raise ValueError("--distance-batch-size must be non-zero")


def concat_field(traces: List[TraceData], field: str) -> np.ndarray:
    return np.concatenate([getattr(t, field) for t in traces], axis=0)


def load_replaced_experts_json(path: Path, layers: List[int], n_expert: int) -> Dict[int, List[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("replaced experts JSON must be an object")

    layer_obj = raw.get("layers", raw)
    if not isinstance(layer_obj, dict):
        raise ValueError("replaced experts JSON must map layer IDs to expert ID lists")

    out: Dict[int, List[int]] = {}
    for layer in layers:
        key = str(layer)
        v = layer_obj.get(key, layer_obj.get(layer, []))
        if not isinstance(v, list):
            raise ValueError(f"layer {layer} must map to a list")
        ids = sorted(set(int(x) for x in v))
        for eid in ids:
            if eid < 0 or eid >= n_expert:
                raise ValueError(f"layer {layer}: expert id {eid} out of range [0, {n_expert})")
        out[layer] = ids
    return out


def derive_replaced_experts(
    topk_ids: np.ndarray,
    layer_ids: np.ndarray,
    layers: List[int],
    n_expert: int,
    replace_ratio: float,
) -> Dict[int, List[int]]:
    if replace_ratio < 0.0 or replace_ratio >= 1.0:
        raise ValueError("--replace-ratio must be in [0.0, 1.0)")

    usage = np.zeros((max(layers) + 1, n_expert), dtype=np.int64)
    for layer in layers:
        rows = np.where(layer_ids == layer)[0]
        if rows.size == 0:
            continue
        flat = topk_ids[rows].reshape(-1)
        binc = np.bincount(flat, minlength=n_expert)
        usage[layer, :] = binc[:n_expert]

    replaced: Dict[int, List[int]] = {}
    for layer in layers:
        n_replace = int(math.floor(n_expert * replace_ratio))
        n_replace = min(max(n_replace, 0), max(n_expert - 1, 0))
        if n_replace == 0:
            replaced[layer] = []
            continue

        order = np.argsort(usage[layer], kind="stable")
        replaced[layer] = [int(x) for x in order[:n_replace]]

    return replaced


def make_replaced_mask(replaced: Dict[int, List[int]], n_layer_total: int, n_expert: int) -> np.ndarray:
    mask = np.zeros((n_layer_total, n_expert), dtype=bool)
    for layer, ids in replaced.items():
        if layer < 0 or layer >= n_layer_total:
            continue
        for eid in ids:
            mask[layer, eid] = True
    return mask


def compute_replaced_mass(
    layer_ids: np.ndarray,
    topk_ids: np.ndarray,
    topk_weights: np.ndarray,
    replaced_mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros(layer_ids.shape[0], dtype=np.float32)
    for layer in np.unique(layer_ids):
        layer = int(layer)
        if layer < 0 or layer >= replaced_mask.shape[0]:
            continue
        rows = np.where(layer_ids == layer)[0]
        if rows.size == 0:
            continue

        ids = topk_ids[rows]
        hit = replaced_mask[layer][ids]
        weights = topk_weights[rows]
        out[rows] = np.sum(np.where(hit, weights, 0.0), axis=1, dtype=np.float32)
    return out


def choose_residual_source(args: argparse.Namespace, has_residual_target: bool, has_y_kept: bool) -> str:
    if args.residual_source != "auto":
        return args.residual_source
    if has_residual_target:
        return "residual_target"
    if has_y_kept:
        return "y_full_minus_y_kept"
    return "proxy_replaced_mass_y_full"


def nearest_centroid_assignments(x: np.ndarray, centroids: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        batch_size = x.shape[0]
    out = np.empty(x.shape[0], dtype=np.int32)

    c2 = np.sum(centroids * centroids, axis=1)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = x[start:end]
        x2 = np.sum(xb * xb, axis=1, keepdims=True)
        d = x2 + c2[None, :] - 2.0 * (xb @ centroids.T)
        out[start:end] = np.argmin(d, axis=1)
    return out


def kmeans(
    x: np.ndarray,
    n_clusters: int,
    n_iters: int,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot run k-means on empty input")

    k = min(n_clusters, n)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = x[init_idx].copy()

    for _ in range(max(n_iters, 1)):
        assign = nearest_centroid_assignments(x, centroids, batch_size=batch_size)

        sums = np.zeros_like(centroids)
        counts = np.bincount(assign, minlength=k)
        np.add.at(sums, assign, x)

        empty = np.where(counts == 0)[0]
        if empty.size > 0:
            refill_idx = rng.choice(n, size=empty.size, replace=False)
            centroids[empty] = x[refill_idx]

        non_empty = counts > 0
        centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

    return centroids


def aggregate_residual_table(
    assignments: np.ndarray,
    residuals: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_clusters, residuals.shape[1]), dtype=np.float32)
    counts = np.bincount(assignments, minlength=n_clusters).astype(np.int32)
    np.add.at(sums, assignments, residuals)

    table = np.zeros_like(sums)
    non_empty = counts > 0
    table[non_empty] = sums[non_empty] / counts[non_empty, None]
    return table, counts


def main() -> int:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)

    traces = [load_trace_npz(p) for p in args.input]
    if not traces:
        raise ValueError("no input traces provided")

    n_embd, n_topk, n_expert = infer_dim_consistency(traces)

    layer_ids = concat_field(traces, "layer_ids")
    h_pre_moe = concat_field(traces, "h_pre_moe")
    topk_ids = concat_field(traces, "topk_ids")
    topk_weights = concat_field(traces, "topk_weights")
    y_full = concat_field(traces, "y_full")

    layers = parse_layers_arg(args.layers, layer_ids)
    if not layers:
        raise ValueError("no layers selected")

    n_layer_total = max(int(np.max(layer_ids)) + 1, max(layers) + 1)

    has_residual_target = all(t.residual_target is not None for t in traces)
    has_y_kept = all(t.y_kept is not None for t in traces)

    if args.replaced_experts_json is not None:
        replaced = load_replaced_experts_json(args.replaced_experts_json, layers, n_expert)
    else:
        replaced = derive_replaced_experts(
            topk_ids=topk_ids,
            layer_ids=layer_ids,
            layers=layers,
            n_expert=n_expert,
            replace_ratio=args.replace_ratio,
        )

    replaced_mask = make_replaced_mask(replaced, n_layer_total=n_layer_total, n_expert=n_expert)
    replaced_mass = compute_replaced_mass(
        layer_ids=layer_ids,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        replaced_mask=replaced_mask,
    )

    residual_source = choose_residual_source(args, has_residual_target=has_residual_target, has_y_kept=has_y_kept)
    if residual_source == "residual_target":
        if not has_residual_target:
            raise ValueError("--residual-source=residual_target requested but some traces lack residual_target")
        residuals = np.concatenate([t.residual_target for t in traces if t.residual_target is not None], axis=0)
    elif residual_source == "y_full_minus_y_kept":
        if not has_y_kept:
            raise ValueError("--residual-source=y_full_minus_y_kept requested but some traces lack y_kept")
        y_kept = np.concatenate([t.y_kept for t in traces if t.y_kept is not None], axis=0)
        residuals = y_full - y_kept
    else:
        residuals = y_full * replaced_mass[:, None]
        print(
            "warning: using proxy residual source (router_mass_replaced * y_full) because optional residual arrays are unavailable",
            file=sys.stderr,
        )

    if residuals.shape != y_full.shape:
        raise ValueError("internal error: residual shape mismatch")

    out_arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int32),
        "algorithm": np.asarray("algorithm2_shared_table"),
        "layers": np.asarray(layers, dtype=np.int32),
        "n_layer_total": np.asarray(n_layer_total, dtype=np.int32),
        "n_embd": np.asarray(n_embd, dtype=np.int32),
        "n_expert": np.asarray(n_expert, dtype=np.int32),
        "n_topk": np.asarray(n_topk, dtype=np.int32),
        "replaced_expert_mask": replaced_mask,
    }

    for layer in layers:
        rows = np.where(layer_ids == layer)[0]
        if rows.size == 0:
            print(f"warning: no rows found for selected layer {layer}, skipping", file=sys.stderr)
            continue

        layer_h = h_pre_moe[rows]
        layer_res = residuals[rows]
        layer_mass = replaced_mass[rows]

        if args.kmeans_max_samples_per_layer > 0 and layer_h.shape[0] > args.kmeans_max_samples_per_layer:
            sample_idx = rng.choice(layer_h.shape[0], size=args.kmeans_max_samples_per_layer, replace=False)
            train_h = layer_h[sample_idx]
        else:
            train_h = layer_h

        centroids = kmeans(
            x=train_h,
            n_clusters=args.clusters_per_layer,
            n_iters=args.kmeans_iters,
            rng=rng,
            batch_size=args.distance_batch_size,
        )
        assignments = nearest_centroid_assignments(layer_h, centroids, batch_size=args.distance_batch_size)

        if args.scaling_mode == "router_mass_replaced":
            valid = layer_mass > args.min_replaced_mass
            if np.any(valid):
                norm = layer_res[valid] / layer_mass[valid, None]
                table, counts = aggregate_residual_table(assignments[valid], norm, n_clusters=centroids.shape[0])
            else:
                table = np.zeros((centroids.shape[0], n_embd), dtype=np.float32)
                counts = np.zeros((centroids.shape[0],), dtype=np.int32)
        else:
            table, counts = aggregate_residual_table(assignments, layer_res, n_clusters=centroids.shape[0])

        out_arrays[f"layer_{layer}_centroids"] = centroids.astype(np.float16)
        out_arrays[f"layer_{layer}_residuals"] = table.astype(np.float16)
        out_arrays[f"layer_{layer}_counts"] = counts.astype(np.int32)
        out_arrays[f"layer_{layer}_mean_replaced_mass"] = np.asarray(float(np.mean(layer_mass)), dtype=np.float32)

    meta = {
        "format_version": 1,
        "algorithm": "algorithm2_shared_table",
        "input_traces": [str(p) for p in args.input],
        "layers": layers,
        "residual_source": residual_source,
        "scaling_mode": args.scaling_mode,
        "clusters_per_layer": int(args.clusters_per_layer),
        "kmeans_iters": int(args.kmeans_iters),
        "kmeans_max_samples_per_layer": int(args.kmeans_max_samples_per_layer),
        "replace_ratio": float(args.replace_ratio),
        "replace_heuristic": args.replace_heuristic,
        "seed": int(args.seed),
        "n_layer_total": int(n_layer_total),
        "n_embd": int(n_embd),
        "n_expert": int(n_expert),
        "n_topk": int(n_topk),
    }
    out_arrays["metadata_json"] = np.asarray(json.dumps(meta, separators=(",", ":")))
    out_arrays["scaling_mode"] = np.asarray(args.scaling_mode)
    out_arrays["residual_source"] = np.asarray(residual_source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out_arrays)

    replaced_out = args.output_replaced_experts
    if replaced_out is None:
        replaced_out = args.output.with_suffix(".replaced-experts.json")
    replaced_out.parent.mkdir(parents=True, exist_ok=True)
    replaced_payload = {
        "format_version": 1,
        "n_layer_total": int(n_layer_total),
        "n_expert": int(n_expert),
        "layers": {str(layer): replaced.get(layer, []) for layer in layers},
    }
    replaced_out.write_text(json.dumps(replaced_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote lookup sidecar: {args.output}")
    print(f"wrote replaced experts: {replaced_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
