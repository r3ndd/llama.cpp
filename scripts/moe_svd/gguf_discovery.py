from __future__ import annotations

import re
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

    for tensor in reader.tensors:
        name = tensor.name
        shape = tuple(int(x) for x in reversed(tensor.shape.tolist()))

        if include_patterns and not _matches_any(name, include_patterns):
            skipped.append(SkippedTensor(name=name, reason="include_filter_no_match"))
            continue

        if exclude_patterns and _matches_any(name, exclude_patterns):
            skipped.append(SkippedTensor(name=name, reason="excluded_by_pattern"))
            continue

        if len(shape) != 2:
            skipped.append(SkippedTensor(name=name, reason="non_2d_tensor"))
            continue

        lowered = name.lower()
        if "expert" not in lowered and "experts" not in lowered and "exp" not in lowered:
            skipped.append(SkippedTensor(name=name, reason="not_expert_pattern"))
            continue

        candidates.append(
            MatrixRef(
                tensor_name=name,
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
        "expert_count": _get_field_value(reader, "llama.expert_count"),
        "expert_used_count": _get_field_value(reader, "llama.expert_used_count"),
    }

    candidates.sort(key=lambda m: (m.layer if m.layer is not None else -1, m.expert if m.expert is not None else -1, m.tensor_name))

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


def load_matrix_by_name(gguf_path: str, tensor_name: str, dtype: str) -> Any:
    reader = open_gguf_reader(gguf_path)
    return load_matrix_from_reader(reader=reader, tensor_name=tensor_name, dtype=dtype)


def open_gguf_reader(gguf_path: str) -> Any:
    gguf = _load_gguf_module()
    path = Path(gguf_path)
    if not path.is_file():
        raise DiscoveryError(f"GGUF file not found: {gguf_path}")

    try:
        return gguf.GGUFReader(str(path), mode="r")
    except Exception as exc:
        raise DiscoveryError(f"Failed to read GGUF file '{gguf_path}': {exc}") from exc


def load_matrix_from_reader(reader: Any, tensor_name: str, dtype: str) -> Any:
    gguf = _load_gguf_module()

    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise DiscoveryError(f"Tensor not found in GGUF: {tensor_name}")

    try:
        matrix = gguf.dequantize(tensor.data, tensor.tensor_type)
    except Exception as exc:
        raise DiscoveryError(
            f"Failed to dequantize tensor '{tensor_name}': {exc}",
        ) from exc
    if matrix.ndim != 2:
        raise DiscoveryError(
            f"Tensor '{tensor_name}' dequantized to non-2D shape {matrix.shape}",
        )

    if dtype == "float32":
        return matrix.astype("float32", copy=False)
    return matrix.astype("float64", copy=False)
