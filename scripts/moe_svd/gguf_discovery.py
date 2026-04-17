from __future__ import annotations

import re
from numbers import Integral
from pathlib import Path
from typing import Any

from .gguf_runtime import ensure_gguf_import
from .types import DiscoveryResult, MatrixRef, SkippedTensor

_LAYER_PATTERNS = (
    re.compile(r"(?:^|[._])(blk|block|layer|layers)[._](\d+)(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])h[._](\d+)(?:[._]|$)", re.IGNORECASE),
)

_EXPERT_PATTERNS = (
    re.compile(r"(?:^|[._])experts?[._](\d+)(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])exp(?:erts?)?[._]?(\d+)(?:[._]|$)", re.IGNORECASE),
)

_ROLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|[._])w1(?:[._]|$)", re.IGNORECASE), "w1"),
    (re.compile(r"(?:^|[._])w2(?:[._]|$)", re.IGNORECASE), "w2"),
    (re.compile(r"(?:^|[._])w3(?:[._]|$)", re.IGNORECASE), "w3"),
    (re.compile(r"gate", re.IGNORECASE), "gate"),
    (re.compile(r"up", re.IGNORECASE), "up"),
    (re.compile(r"down", re.IGNORECASE), "down"),
)


class DiscoveryError(RuntimeError):
    """Raised when GGUF parsing/discovery fails."""


def _extract_layer(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match is None:
            continue
        group = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
        try:
            return int(group)
        except ValueError:
            return None
    return None


def _extract_expert(name: str) -> int | None:
    for pattern in _EXPERT_PATTERNS:
        match = pattern.search(name)
        if match is None:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_role(name: str) -> str | None:
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(name):
            return role
    return None


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error as exc:
            raise DiscoveryError(f"Invalid regex pattern '{pat}': {exc}") from exc
    return compiled


def _matches_any(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(name) for p in patterns)


def _is_expert_like_name(name: str) -> bool:
    lowered = name.lower()
    return "expert" in lowered or "experts" in lowered or "exp" in lowered


def _choose_packed_expert_axis(
    *,
    shape: tuple[int, int, int],
    expert_count_hint: int | None,
) -> int | None:
    if expert_count_hint is not None:
        matching_axes = [axis for axis, dim in enumerate(shape) if dim == expert_count_hint]
        if len(matching_axes) == 1:
            return matching_axes[0]

    return None


def _unpack_packed_expert_tensor(
    *,
    tensor_name: str,
    shape: tuple[int, int, int],
    tensor_type: str,
    expert_count_hint: int | None,
) -> tuple[list[MatrixRef], str | None]:
    expert_axis = _choose_packed_expert_axis(
        shape=shape,
        expert_count_hint=expert_count_hint,
    )
    if expert_axis is None:
        return [], "packed_expert_tensor_3d_unknown_layout"

    expert_dim = shape[expert_axis]
    matrix_shape = tuple(dim for axis, dim in enumerate(shape) if axis != expert_axis)
    if len(matrix_shape) != 2:
        return [], "packed_expert_tensor_3d_invalid_matrix_shape"

    layer = _extract_layer(tensor_name)
    role = _extract_role(tensor_name)

    unpacked: list[MatrixRef] = []
    for expert_idx in range(expert_dim):
        unpacked.append(
            MatrixRef(
                tensor_name=f"{tensor_name}::expert[{expert_idx}]",
                source_tensor_name=tensor_name,
                shape=(matrix_shape[0], matrix_shape[1]),
                layer=layer,
                expert=expert_idx,
                role=role,
                tensor_type=tensor_type,
                packed_expert_index=expert_idx,
                packed_expert_axis=expert_axis,
            ),
        )

    return unpacked, None


def _load_gguf_module() -> Any:
    try:
        return ensure_gguf_import()
    except Exception as exc:
        raise DiscoveryError(
            "Failed to import gguf Python module. Ensure gguf-py is available on PYTHONPATH.",
        ) from exc


def discover_expert_matrices(
    gguf_path: str,
    include: list[str],
    exclude: list[str],
) -> DiscoveryResult:
    reader = open_gguf_reader(gguf_path)

    include_patterns = _compile_patterns(include)
    exclude_patterns = _compile_patterns(exclude)

    path = Path(gguf_path)
    if not path.is_file():
        raise DiscoveryError(f"GGUF file not found: {gguf_path}")

    candidates: list[MatrixRef] = []
    skipped: list[SkippedTensor] = []
    expert_count_hint_raw = _get_arch_or_llama_field(reader, "expert_count")
    expert_count_hint = int(expert_count_hint_raw) if isinstance(expert_count_hint_raw, Integral) else None

    for tensor in reader.tensors:
        name = tensor.name
        shape = tuple(int(x) for x in reversed(tensor.shape.tolist()))
        expert_like = _is_expert_like_name(name)

        if include_patterns and not _matches_any(name, include_patterns):
            skipped.append(SkippedTensor(name=name, reason="include_filter_no_match"))
            continue

        if exclude_patterns and _matches_any(name, exclude_patterns):
            skipped.append(SkippedTensor(name=name, reason="excluded_by_pattern"))
            continue

        if len(shape) == 3 and expert_like:
            unpacked, skip_reason = _unpack_packed_expert_tensor(
                tensor_name=name,
                shape=(shape[0], shape[1], shape[2]),
                tensor_type=getattr(tensor.tensor_type, "name", str(tensor.tensor_type)),
                expert_count_hint=expert_count_hint,
            )
            if unpacked:
                candidates.extend(unpacked)
            elif skip_reason is not None:
                skipped.append(SkippedTensor(name=name, reason=skip_reason))
            continue

        if len(shape) != 2:
            skipped.append(SkippedTensor(name=name, reason="non_2d_tensor"))
            continue

        if not expert_like:
            skipped.append(SkippedTensor(name=name, reason="not_expert_pattern"))
            continue

        candidates.append(
            MatrixRef(
                tensor_name=name,
                source_tensor_name=name,
                shape=(shape[0], shape[1]),
                layer=_extract_layer(name),
                expert=_extract_expert(name),
                role=_extract_role(name),
                tensor_type=getattr(tensor.tensor_type, "name", str(tensor.tensor_type)),
            ),
        )

    metadata = {
        "gguf_path": str(path.resolve()),
        "architecture": _get_field_value(reader, "general.architecture"),
        "name": _get_field_value(reader, "general.name"),
        "file_type": _get_field_value(reader, "general.file_type"),
        "block_count": _get_arch_or_llama_field(reader, "block_count"),
        "expert_count": expert_count_hint,
        "expert_used_count": _get_arch_or_llama_field(reader, "expert_used_count"),
    }

    candidates.sort(
        key=lambda m: (
            m.layer if m.layer is not None else -1,
            m.source_tensor_name,
            m.expert if m.expert is not None else -1,
            m.tensor_name,
        ),
    )

    return DiscoveryResult(
        total_tensors=len(reader.tensors),
        candidates=candidates,
        skipped=skipped,
        metadata=metadata,
    )


def _get_field_value(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    try:
        return field.contents()
    except Exception:
        return None


def _get_arch_or_llama_field(reader: Any, suffix: str) -> Any:
    architecture = _get_field_value(reader, "general.architecture")

    keys: list[str] = []
    if isinstance(architecture, str) and architecture:
        keys.append(f"{architecture}.{suffix}")
    keys.append(f"llama.{suffix}")

    for key in keys:
        value = _get_field_value(reader, key)
        if value is not None:
            return value

    return None


def load_matrix_by_name(gguf_path: str, tensor_name: str, dtype: str) -> Any:
    reader = open_gguf_reader(gguf_path)
    matrix_ref = MatrixRef(
        tensor_name=tensor_name,
        source_tensor_name=tensor_name,
        shape=(0, 0),
        layer=None,
        expert=None,
        role=None,
        tensor_type="unknown",
    )
    return load_matrix_from_reader(reader=reader, matrix_ref=matrix_ref, dtype=dtype)


def open_gguf_reader(gguf_path: str) -> Any:
    gguf = _load_gguf_module()
    path = Path(gguf_path)
    if not path.is_file():
        raise DiscoveryError(f"GGUF file not found: {gguf_path}")

    try:
        return gguf.GGUFReader(str(path), mode="r")
    except Exception as exc:
        raise DiscoveryError(f"Failed to read GGUF file '{gguf_path}': {exc}") from exc


def load_matrix_from_reader(reader: Any, matrix_ref: MatrixRef, dtype: str) -> Any:
    gguf = _load_gguf_module()

    source_tensor_name = matrix_ref.source_tensor_name
    tensor = next((t for t in reader.tensors if t.name == source_tensor_name), None)
    if tensor is None:
        raise DiscoveryError(f"Tensor not found in GGUF: {source_tensor_name}")

    try:
        matrix = gguf.dequantize(tensor.data, tensor.tensor_type)
    except Exception as exc:
        raise DiscoveryError(
            f"Failed to dequantize tensor '{source_tensor_name}': {exc}",
        ) from exc

    if matrix_ref.packed_expert_index is None:
        if matrix.ndim != 2:
            raise DiscoveryError(
                f"Tensor '{source_tensor_name}' dequantized to non-2D shape {matrix.shape}",
            )
        matrix_2d = matrix
    else:
        if matrix.ndim != 3:
            raise DiscoveryError(
                f"Packed tensor '{source_tensor_name}' dequantized to non-3D shape {matrix.shape}",
            )
        axis = matrix_ref.packed_expert_axis
        expert_idx = matrix_ref.packed_expert_index
        if axis is None or expert_idx is None:
            raise DiscoveryError(f"Packed tensor '{source_tensor_name}' missing unpack metadata")
        if axis < 0 or axis >= matrix.ndim:
            raise DiscoveryError(
                f"Packed tensor '{source_tensor_name}' has invalid expert axis {axis} for shape {matrix.shape}",
            )
        if expert_idx < 0 or expert_idx >= matrix.shape[axis]:
            raise DiscoveryError(
                f"Packed tensor '{source_tensor_name}' expert index {expert_idx} out of bounds for shape {matrix.shape}",
            )

        matrix_2d = matrix.take(indices=expert_idx, axis=axis)
        if matrix_2d.ndim != 2:
            raise DiscoveryError(
                f"Packed tensor '{source_tensor_name}' expert slice {expert_idx} yielded non-2D shape {matrix_2d.shape}",
            )

    if matrix_ref.shape != (0, 0):
        actual_shape = (int(matrix_2d.shape[0]), int(matrix_2d.shape[1]))
        if actual_shape != matrix_ref.shape:
            raise DiscoveryError(
                f"Tensor '{source_tensor_name}' produced shape {actual_shape}, expected {matrix_ref.shape}",
            )

    if dtype == "float32":
        return matrix_2d.astype("float32", copy=False)
    return matrix_2d.astype("float64", copy=False)
