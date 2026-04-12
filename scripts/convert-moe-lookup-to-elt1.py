#!/usr/bin/env python3

"""Convert MoE lookup NPZ sidecars to ELT1 runtime binary sidecars.

Input NPZ is produced by scripts/build-moe-lookup.py.
Output is ELT1 binary payload consumed by src/moe-lookup.cpp.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ELT1_MAGIC = 0x31544C45  # "ELT1" little-endian u32
ELT1_FORMAT_VERSION = 1
ELT1_VECTOR_DTYPE_FP16 = 1
ELT1_SCALING_S_MISSING = 1
MAX_MODEL_ID_LEN = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert build-moe-lookup NPZ output to ELT1 runtime binary sidecar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input lookup NPZ from build-moe-lookup.py")
    parser.add_argument(
        "--replaced-experts",
        type=Path,
        required=True,
        help="Input replaced-experts JSON from build-moe-lookup.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output ELT1 sidecar path")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Runtime model ID string; must match model.arch_name() at load time",
    )
    return parser.parse_args()


def _require_npz_key(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz:
        raise ValueError(f"missing required NPZ key '{key}'")
    return npz[key]


def _read_int_scalar(npz: np.lib.npyio.NpzFile, key: str) -> int:
    arr = _require_npz_key(npz, key)
    if arr.shape != ():
        raise ValueError(f"NPZ key '{key}' must be a scalar")
    try:
        value = int(arr)
    except Exception as exc:
        raise ValueError(f"NPZ key '{key}' is not an integer scalar") from exc
    return value


def _validate_model_id(model_id: str) -> bytes:
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) == 0:
        raise ValueError("--model-id must not be empty")
    if len(model_id_bytes) > MAX_MODEL_ID_LEN:
        raise ValueError(f"--model-id UTF-8 byte length exceeds {MAX_MODEL_ID_LEN}")
    return model_id_bytes


def _normalize_scaling_mode(npz: np.lib.npyio.NpzFile) -> str:
    if "scaling_mode" in npz:
        mode = str(npz["scaling_mode"])
    elif "runtime_scaling" in npz:
        mode = str(npz["runtime_scaling"])
    else:
        raise ValueError("missing required NPZ key 'scaling_mode' or 'runtime_scaling'")

    if mode == "router_mass_replaced":
        mode = "s_missing"

    if mode != "s_missing":
        raise ValueError(f"unsupported scaling mode '{mode}' for ELT1 (must be 's_missing')")

    return mode


def _load_replaced_json(path: Path, n_layer: int, n_expert: int) -> Dict[int, List[int]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"replaced experts JSON does not exist: {path}") from exc
    except Exception as exc:
        raise ValueError(f"failed to read replaced experts JSON '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("replaced experts JSON must be an object")

    if "format_version" in raw and int(raw["format_version"]) != 1:
        raise ValueError(f"unsupported replaced experts format_version={raw['format_version']}, expected 1")

    if "n_layer_total" in raw and int(raw["n_layer_total"]) != n_layer:
        raise ValueError(
            f"replaced experts n_layer_total mismatch ({int(raw['n_layer_total'])} vs NPZ n_layer_total={n_layer})"
        )

    if "n_expert" in raw and int(raw["n_expert"]) != n_expert:
        raise ValueError(f"replaced experts n_expert mismatch ({int(raw['n_expert'])} vs NPZ n_expert={n_expert})")

    layer_obj = raw.get("layers", raw)
    if not isinstance(layer_obj, dict):
        raise ValueError("replaced experts JSON must map layer IDs to expert ID lists")

    parsed: Dict[int, List[int]] = {}
    for layer_key, raw_ids in layer_obj.items():
        try:
            layer_id = int(layer_key)
        except Exception as exc:
            raise ValueError(f"invalid layer id key in replaced experts JSON: {layer_key!r}") from exc

        if layer_id < 0 or layer_id >= n_layer:
            raise ValueError(f"replaced experts layer id {layer_id} out of range [0, {n_layer})")

        if not isinstance(raw_ids, list):
            raise ValueError(f"layer {layer_id}: replaced expert IDs must be a list")

        parsed_ids = [int(x) for x in raw_ids]
        if len(set(parsed_ids)) != len(parsed_ids):
            raise ValueError(f"layer {layer_id}: duplicate replaced expert IDs are not allowed")

        uniq_sorted = sorted(parsed_ids)
        for eid in uniq_sorted:
            if eid < 0 or eid >= n_expert:
                raise ValueError(f"layer {layer_id}: expert id {eid} out of range [0, {n_expert})")
        parsed[layer_id] = uniq_sorted

    return parsed


def _parse_layers(npz: np.lib.npyio.NpzFile, n_layer: int) -> List[int]:
    layers = np.asarray(_require_npz_key(npz, "layers"), dtype=np.int32)
    if layers.ndim != 1:
        raise ValueError("NPZ key 'layers' must be 1D")
    if layers.size == 0:
        raise ValueError("NPZ key 'layers' must not be empty")

    out: List[int] = [int(x) for x in layers.tolist()]
    if len(set(out)) != len(out):
        raise ValueError("NPZ key 'layers' contains duplicate layer IDs")

    for layer_id in out:
        if layer_id < 0 or layer_id >= n_layer:
            raise ValueError(f"layer id {layer_id} out of range [0, {n_layer})")

    return out


def convert(npz_path: Path, replaced_path: Path, output_path: Path, model_id: str) -> None:
    model_id_bytes = _validate_model_id(model_id)

    try:
        npz = np.load(npz_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise ValueError(f"input NPZ does not exist: {npz_path}") from exc
    except Exception as exc:
        raise ValueError(f"failed to load NPZ '{npz_path}': {exc}") from exc

    with npz:
        fmt = _read_int_scalar(npz, "format_version")
        if fmt != 1:
            raise ValueError(f"unsupported NPZ format_version={fmt}, expected 1")

        _normalize_scaling_mode(npz)

        n_layer = _read_int_scalar(npz, "n_layer_total")
        n_embd = _read_int_scalar(npz, "n_embd")
        n_expert = _read_int_scalar(npz, "n_expert")
        n_expert_used = _read_int_scalar(npz, "n_topk")

        if n_layer <= 0:
            raise ValueError("n_layer_total must be > 0")
        if n_embd <= 0:
            raise ValueError("n_embd must be > 0")
        if n_expert <= 0:
            raise ValueError("n_expert must be > 0")
        if n_expert_used <= 0 or n_expert_used > n_expert:
            raise ValueError("n_topk must be in range [1, n_expert]")

        layers = _parse_layers(npz, n_layer=n_layer)
        replaced_by_layer = _load_replaced_json(replaced_path, n_layer=n_layer, n_expert=n_expert)

        npz_replaced_mask = None
        if "replaced_expert_mask" in npz:
            npz_replaced_mask = np.asarray(npz["replaced_expert_mask"], dtype=bool)
            expected_shape = (n_layer, n_expert)
            if npz_replaced_mask.shape != expected_shape:
                raise ValueError(
                    "NPZ key 'replaced_expert_mask' has invalid shape "
                    f"{npz_replaced_mask.shape}, expected {expected_shape}"
                )

        payloads: List[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        fill_safe_limit = n_expert - n_expert_used

        for layer_id in layers:
            centroids_key = f"layer_{layer_id}_centroids"
            contrib_key = f"layer_{layer_id}_contributions"

            centroids = np.asarray(_require_npz_key(npz, centroids_key), dtype=np.float32)
            contributions = np.asarray(_require_npz_key(npz, contrib_key), dtype=np.float32)

            if centroids.ndim != 2 or contributions.ndim != 2:
                raise ValueError(f"layer {layer_id}: centroids/contributions must be rank-2")
            if centroids.shape != contributions.shape:
                raise ValueError(
                    f"layer {layer_id}: centroids and contributions shape mismatch "
                    f"({centroids.shape} vs {contributions.shape})"
                )

            n_keys, layer_embd = int(centroids.shape[0]), int(centroids.shape[1])
            if n_keys <= 0:
                raise ValueError(f"layer {layer_id}: n_keys must be > 0")
            if layer_embd != n_embd:
                raise ValueError(f"layer {layer_id}: n_embd mismatch ({layer_embd} vs expected {n_embd})")
            if n_keys * n_embd > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"layer {layer_id}: n_keys*n_embd={n_keys * n_embd} exceeds uint32 payload bound"
                )

            if not np.all(np.isfinite(centroids)):
                raise ValueError(f"layer {layer_id}: centroid contains non-finite value")
            if not np.all(np.isfinite(contributions)):
                raise ValueError(f"layer {layer_id}: contribution contains non-finite value")

            fp16_max = np.finfo(np.float16).max
            if np.max(np.abs(centroids)) > fp16_max:
                raise ValueError(f"layer {layer_id}: centroid overflows fp16")
            if np.max(np.abs(contributions)) > fp16_max:
                raise ValueError(f"layer {layer_id}: contribution overflows fp16")

            replaced_ids = replaced_by_layer.get(layer_id, [])
            if len(replaced_ids) > fill_safe_limit:
                raise ValueError(
                    f"layer {layer_id}: replaced_count={len(replaced_ids)} exceeds fill-safe limit {fill_safe_limit}"
                )

            if npz_replaced_mask is not None:
                mask_ids = np.flatnonzero(npz_replaced_mask[layer_id]).astype(np.int32).tolist()
                if mask_ids != replaced_ids:
                    raise ValueError(
                        f"layer {layer_id}: replaced experts mismatch between NPZ mask and JSON "
                        f"({mask_ids} vs {replaced_ids})"
                    )

            centroids_f16 = centroids.astype(np.dtype("<f2"), copy=False)
            contrib_f16 = contributions.astype(np.dtype("<f2"), copy=False)

            replaced_u32 = np.asarray(replaced_ids, dtype=np.dtype("<u4"))

            payloads.append((layer_id, centroids_f16, contrib_f16, replaced_u32))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_bytes = struct.pack(
        "<10I",
        ELT1_MAGIC,
        ELT1_FORMAT_VERSION,
        len(model_id_bytes),
        n_layer,
        n_embd,
        n_expert,
        n_expert_used,
        ELT1_VECTOR_DTYPE_FP16,
        ELT1_SCALING_S_MISSING,
        len(payloads),
    )

    with output_path.open("wb") as out:
        out.write(header_bytes)
        out.write(model_id_bytes)

        for layer_id, centroids_f16, contrib_f16, replaced_u32 in payloads:
            layer_header = struct.pack(
                "<3I",
                layer_id,
                int(centroids_f16.shape[0]),
                int(replaced_u32.shape[0]),
            )
            out.write(layer_header)
            out.write(centroids_f16.tobytes(order="C"))
            out.write(contrib_f16.tobytes(order="C"))
            out.write(replaced_u32.tobytes(order="C"))


def main() -> int:
    args = parse_args()
    convert(args.input, args.replaced_experts, args.output, args.model_id)
    print(f"wrote ELT1 sidecar: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
